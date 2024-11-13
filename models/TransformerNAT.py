import torch
import torch.nn as nn
import torch.nn.functional as F

from models.layers.Attention import FullAttention, AttentionLayer
from models.layers.Embed import DataEmbedding, ValueEmbedding
from models.layers.Transformer_EncDec import (
    EncoderLayer,
    DecoderLayer,
    Encoder,
    Decoder,
)


class Model(nn.Module):
    """
    Vanilla Transformer with O(L^2) complexity
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        if configs.features in ["S", "MS"]:
            print(f"To predict single variable, extra output linear dimension is not required, set to 1!")
            configs.c_out = 1
        self.label_len = configs.label_len
        self.pred_len = configs.pred_len
        self.encoder_regular = configs.encoder_regular
        self.output_attention = configs.output_attention

        # Embedding
        self.enc_embedding = DataEmbedding(
            configs.enc_in, configs.d_model, configs.embed, configs.freq, configs.dropout
        )
        if hasattr(configs, "embed_shared") and configs.embed_shared:
            self.dec_embedding = self.enc_embedding
        else:
            self.dec_embedding = DataEmbedding(
                configs.dec_in, configs.d_model, configs.embed, configs.freq, configs.dropout
            )
        # Encoder
        self.encoder = Encoder(
            attn_layers=[
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(
                            False, configs.factor, attention_dropout=configs.dropout, output_attention=False),
                        configs.d_model,
                        configs.n_heads
                    ),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for _ in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )
        # Decoder
        self.decoder = Decoder(
            layers=[
                DecoderLayer(
                    # self-attn, w/o triangular causal mask for nat variant
                    AttentionLayer(
                        FullAttention(
                            mask_flag=False, attention_dropout=configs.dropout, output_attention=False
                        ),
                        configs.d_model,
                        configs.n_heads
                    ),
                    # cross-attn,
                    AttentionLayer(
                        FullAttention(
                            mask_flag=False, attention_dropout=configs.dropout, output_attention=False
                        ),
                        configs.d_model,
                        configs.n_heads
                    ),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation,
                )
                for _ in range(configs.d_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model),
        )
        if configs.conv_head:
            from models.layers.Embed import ValueEmbedding
            self.predict_head = ValueEmbedding(configs.d_model, configs.c_out)
        else:
            self.predict_head = nn.Linear(configs.d_model, configs.c_out, bias=True)
        self.loginfo = self._set_loginfo_(configs)

    def _set_loginfo_(self, cfg):

        return f"_s-{cfg.seq_len}_l-{cfg.label_len}_p-{cfg.pred_len}" \
               f"-f-{cfg.freq}" \
               f"_embed-{cfg.embed}{'-shrd' if cfg.embed_shared else ''}" \
               f"{'_rgl' if cfg.encoder_regular else ''}" \
               f"_ec-{cfg.enc_in}_dc-{cfg.dec_in}_oc-{cfg.c_out}" \
               f"_el-{cfg.e_layers}_dl-{cfg.d_layers}_d-{cfg.d_model}_nh-{cfg.n_heads}_ff-{cfg.d_ff}" \
               f"_drop-{cfg.dropout}"

    def get_loginfo(self):
        return self.loginfo

    def forward(
            self,
            prev_x,
            prev_y,
            prev_mark,
            target_x,
            target_y,
            target_mark,
            enc_self_mask=None,
            dec_self_mask=None,
            dec_enc_mask=None,
            **kwargs,
    ):
        # prepare decoder input
        # dec_inp = torch.zeros_like(target_y)
        if self.label_len != 0:
            target_x = torch.cat([prev_x[:, -self.label_len:, :], target_x], dim=1)
            target_mark = torch.cat([prev_mark[:, -self.label_len:, :], target_mark], dim=1)

        # if self.revin:
        #     z = z.permute(0, 2, 1)
        #     z = self.revin_layer(z, 'norm')
        #     z = z.permute(0, 2, 1)

        """
        B, T_target, D = target_x.shape
        mask_shape = [B, 1, T_target, T_target]
        # mask = torch.triu(torch.ones(mask_shape, dtype=torch.bool), diagonal=1+T_target).to(prev_x.device)
        cross_mask = torch.zeros(mask_shape)
        stride = 10
        for i in range(T_target//stride):
            inx = i * stride
            cross_mask[..., inx:inx + stride, inx:inx + stride] = False
        cross_mask[..., T_target // stride * stride:, :] = False
        """

        enc_embed = self.enc_embedding.forward(prev_x, prev_mark)
        enc_out_dict = self.encoder.forward(enc_embed)
        dec_embed = self.dec_embedding(target_x, target_mark)
        dec_out_dict = self.decoder(dec_embed, enc_out_dict['output'])
        output = self.predict_head(dec_out_dict['output'])
        output_dict = {
            "output": output[:, -self.pred_len:, :],
            "encoder_output": enc_out_dict['output'],
            "encoder_attns": enc_out_dict['attns'],
            "decoder_output": dec_out_dict['output'],
        }
        if self.encoder_regular > 0:
            enc_mse_loss = F.mse_loss(self.predict_head(enc_out_dict['output']), prev_y) * self.encoder_regular
            output_dict.update({"encoder_mse_loss": enc_mse_loss})
        return output_dict
