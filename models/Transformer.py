import torch
import torch.nn as nn

from models.layers.Attention import FullAttention, AttentionLayer
from models.layers.Embed import DataEmbedding
from models.layers.Transformer_EncDec import Decoder, DecoderLayer, Encoder, EncoderLayer


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
        self.output_attention = configs.output_attention

        # Embedding
        self.enc_embedding = DataEmbedding(
            configs.enc_in, configs.d_model, configs.embed, configs.freq, configs.dropout
        )
        self.dec_embedding = DataEmbedding(
            configs.dec_in, configs.d_model, configs.embed, configs.freq, configs.dropout
        )
        # Encoder
        self.encoder = Encoder(
            [
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
            [
                DecoderLayer(
                    # self-attn
                    AttentionLayer(
                        FullAttention(True, configs.factor, attention_dropout=configs.dropout, output_attention=False),
                        configs.d_model,
                        configs.n_heads
                    ),
                    # cross-attn
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout, output_attention=False),
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
            projection=nn.Linear(configs.d_model, configs.c_out, bias=True)
        )
        self.loginfo = self._set_loginfo_(configs)

    def _set_loginfo_(self, cfg):

        return f"_sl_{cfg.seq_len}_ll_{cfg.label_len}_pl_{cfg.pred_len}" \
               f"-freq_{cfg.freq}" \
               f"_embed_{cfg.embed}_encin_{cfg.enc_in}_decin_{cfg.dec_in}_deout_{cfg.c_out}" \
               f"_el_{cfg.e_layers}_dl_{cfg.d_layers}_dmodel_{cfg.d_model}_nhead_{cfg.n_heads}_dff_{cfg.d_ff}" \
               f"_dropout_{cfg.dropout}"

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

        enc_embed = self.enc_embedding.forward(prev_x, prev_mark)
        enc_out_dict = self.encoder.forward(enc_embed, attn_mask=enc_self_mask)

        dec_embed = self.dec_embedding(target_x, target_mark)
        dec_out_dict = self.decoder(
            dec_embed, enc_out_dict['output'],
            x_mask=dec_self_mask, cross_mask=dec_enc_mask
        )
        output = dec_out_dict['output'][:, -self.pred_len:, :]
        return {
            # decoder output
            "output": output,
            "decoder_output": dec_out_dict['output'],
            # "decoder_attns": dec_out_dict['attns'],
            # "decoder_sub_disagreements": dec_out_dict['sub_disagreements'],
            # "decoder_out_disagreements": dec_out_dict['out_disagreements'],
            # "decoder_cross_attns": dec_out_dict['cross_attns'],
            # "decoder_cross_sub_disagreements": dec_out_dict['cross_sub_disagreements'],
            # "decoder_cross_out_disagreements": dec_out_dict['cross_out_disagreements'],
            # encoder output
            "encoder_output": enc_out_dict['output'],
            "encoder_attns": enc_out_dict['attns'],
            # "encoder_sub_disagreements": enc_out_dict['sub_disagreements'],
            # "encoder_out_disagreements": enc_out_dict['out_disagreements'],
        }
