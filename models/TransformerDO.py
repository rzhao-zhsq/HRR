import torch
import torch.nn as nn
import torch.nn.functional as F

from models.layers.Attention import FullAttention, AttentionLayer
from models.layers.Embed import DataEmbedding
from models.layers.Transformer_EncDec import (
    EncoderLayer,
    Encoder,
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
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.encoder_regular = configs.encoder_regular
        self.output_attention = configs.output_attention

        # Embedding
        self.embedding = DataEmbedding(
            configs.enc_in, configs.d_model, configs.embed, configs.freq, configs.dropout
        )

        # Encoder
        self.decoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(
                            mask_flag=True, attention_dropout=configs.dropout, output_attention=False
                        ),
                        configs.d_model,
                        configs.n_heads
                    ),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for _ in range(configs.d_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )

        if configs.conv_head:
            from models.layers.Embed import ValueEmbedding
            self.predict_head = ValueEmbedding(configs.d_model, configs.c_out)
        else:
            self.predict_head = nn.Linear(configs.d_model, configs.c_out, bias=True)

        self.loginfo = self._set_loginfo_(configs)

    def _set_loginfo_(self, cfg):

        return f"_s-{cfg.seq_len}_p-{cfg.pred_len}" \
               f"-f-{cfg.freq}" \
               f"_embed-{cfg.embed}" \
               f"{'_rgl' if cfg.encoder_regular else ''}" \
               f"_dc-{cfg.dec_in}_oc-{cfg.c_out}" \
               f"_dl-{cfg.d_layers}_d-{cfg.d_model}_nh-{cfg.n_heads}_ff-{cfg.d_ff}" \
               f"_drop-{cfg.dropout}"

    def get_loginfo(self):
        return self.loginfo

    # TODO(rzhao): auto-regressive decode
    def decode(self):
        return

    def forward(
            self,
            prev_x,
            prev_y,
            prev_mark,
            target_x,
            target_y,
            target_mark,
            dec_self_mask=None,
            **kwargs,
    ):
        # prepare decoder input
        # dec_inp = torch.zeros_like(target_y)
        # if self.label_len != 0:
        #     target_x = torch.cat([prev_x[:, -self.label_len:, :], target_x], dim=1)
        #     target_mark = torch.cat([prev_mark[:, -self.label_len:, :], target_mark], dim=1)

        B, T_prev, D = prev_x.shape
        B, T_target, D = target_x.shape
        mask_shape = [B, 1, T_prev + T_target, T_prev + T_target]
        mask = torch.triu(torch.ones(mask_shape, dtype=torch.bool), diagonal=1+T_target).to(prev_x.device)

        input_x = torch.cat([prev_x, target_x], dim=1)
        mark = torch.cat([prev_mark, target_mark], dim=1)

        embed = self.embedding.forward(input_x, mark)
        out_dict = self.decoder.forward(embed, attn_mask=mask)

        output = self.predict_head(out_dict['output'])
        output = output
        output_dict = {
            "output": output[:, -self.pred_len:, :],
            "encoder_output": out_dict['output'],
            "encoder_attns": out_dict['attns'],
        }
        if self.encoder_regular:
            enc_mse_loss = F.mse_loss(output[:, :self.seq_len, :], prev_y) * self.encoder_regular
            output_dict.update({"encoder_mse_loss": enc_mse_loss})
        return output_dict
