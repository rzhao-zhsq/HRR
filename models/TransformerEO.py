import torch
import torch.nn as nn

from models.layers.Attention import FullAttention, AttentionLayer
from models.layers.Embed import DataEmbedding
from models.layers.Transformer_EncDec import Decoder, DecoderLayer, Encoder, EncoderLayer
from models.layers.RevIN import RevIN


class Model(nn.Module):
    """
    Vanilla Transformer with O(L^2) complexity, Encoder Only version.
    """
    def __init__(self, configs):
        super(Model, self).__init__()
        if configs.features in ["S", "MS"]:
            print(f"To predict single variable, extra output linear dimension is not required, set to 1!")
            configs.c_out = 1
        self.output_attention = configs.output_attention
        # pdb.trace()
        assert configs.seq_len == configs.pred_len
        self.pred_len = configs.pred_len
        # RevIn
        self.revin = configs.revin
        if self.revin:
            self.revin_layer = RevIN(
                num_features=configs.enc_in,
                affine=configs.affine if hasattr(configs, "affine") else False,
            )

        # Embedding
        self.enc_embedding = DataEmbedding(
            c_in=configs.enc_in,
            d_model=configs.d_model,
            embed_type=configs.embed,
            freq=configs.freq,
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
        self.pred_head = nn.Linear(configs.d_model, configs.c_out, bias=True)

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
        if self.revin:
            target_x = self.revin_layer(target_x, 'norm')

        enc_embed = self.enc_embedding(target_x, target_mark)
        enc_out_dict = self.encoder(enc_embed, attn_mask=enc_self_mask)
        output = self.pred_head(enc_out_dict['output'][:, -self.pred_len:, :])

        if self.revin:
            output = self.revin_layer(output, 'denorm')

        return {
            "output": output,
            # encoder output
            "encoder_output": enc_out_dict['output'],
            "encoder_attns": enc_out_dict['attns'],
        }

