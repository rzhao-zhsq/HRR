import torch
import torch.nn as nn
import torch.nn.functional as F

from models.layers.Embed import DataEmbedding_inverted
from models.layers.Attention import FullAttention, AttentionLayer
from models.layers.Transformer_EncDec import Encoder, EncoderLayer


class Model(nn.Module):
    """
    Paper link: https://arxiv.org/abs/2310.06625
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention
        # Embedding
        self.enc_embedding = DataEmbedding_inverted(
            c_in=configs.seq_len,
            d_model=configs.d_model,
            dropout=configs.dropout,
        )
        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(
                            False, configs.factor, attention_dropout=configs.dropout, output_attention=False
                        ),
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
        self.projection = nn.Linear(configs.d_model, configs.pred_len, bias=True)

        self.loginfo = self._set_loginfo_(configs)

    def _set_loginfo_(self, cfg):

        return f"_sl_{cfg.seq_len}_ll_{cfg.label_len}_pl_{cfg.pred_len}" \
               f"-freq_{cfg.freq}" \
               f"-revin_{cfg.revin}" \
               f"_embed_{cfg.embed}_cin_{cfg.enc_in}_cout{cfg.c_out}" \
               f"_el_{cfg.e_layers}_dmodel_{cfg.d_model}_nhead_{cfg.n_heads}_dff_{cfg.d_ff}" \
               f"_dropout_{cfg.dropout}"

    def get_loginfo(self):
        return self.loginfo

    def forward(
            self,
            target_x,
            target_y,
            target_mark,
            enc_self_mask=None,
            **kwargs,
    ):
        # Normalization from Non-stationary Transformer
        means = target_x.mean(1, keepdim=True).detach()
        x_enc = target_x - means
        std = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= std

        _, _, N = x_enc.shape

        # Embedding
        enc_out = self.enc_embedding(x_enc, target_mark)
        enc_out_dict = self.encoder(enc_out, attn_mask=None)

        output = self.projection(
            enc_out_dict['output'][:, -self.pred_len:, :]
        ).permute(0, 2, 1)[:, :, :N]
        # De-Normalization from Non-stationary Transformer
        output = output * (std[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        output = output + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        return {
            "output": output,
            # encoder output
            "encoder_output": enc_out_dict['output'],
            "encoder_attns": enc_out_dict['attns'],
        }

