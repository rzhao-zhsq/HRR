import torch
import torch.nn as nn

from models.layers.Attention import FullAttention, AttentionLayer
from models.layers.Embed import DataEmbedding
from models.layers.Transformer_EncDec import Decoder, DecoderLayer, Encoder, EncoderLayer
from models.layers.RevIN import RevIN
from models.Conv import Conv1d, ConvStack


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
            configs.enc_in,
            configs.d_model,
            configs.embed,
            configs.freq,
            configs.dropout,
            conv=configs.conv_head,
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
        if configs.conv_head:
            self.regression_head = Conv1d(configs.d_model, configs.c_out)
        else:
            self.regression_head = nn.Linear(configs.d_model, configs.c_out, bias=True)

        self.loginfo = self._set_loginfo_(configs)

    def _set_loginfo_(self, cfg):

        return f"_s-{cfg.seq_len}_l-{cfg.label_len}_p-{cfg.pred_len}" \
               f"-freq-{cfg.freq}" \
               f"_embed-{cfg.embed}_cin-{cfg.enc_in}_cout-{cfg.c_out}" \
               f"_el-{cfg.e_layers}_dmodel-{cfg.d_model}_nhead-{cfg.n_heads}_dff-{cfg.d_ff}" \
               f"_dropout-{cfg.dropout}"
        # f"-revin_{cfg.revin}" \

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
        # if self.revin:
        #     target_x = self.revin_layer(target_x, 'norm')

        enc_embed = self.enc_embedding(target_x, target_mark)
        enc_out_dict = self.encoder(enc_embed, attn_mask=enc_self_mask)
        output = self.regression_head(enc_out_dict['output'])

        # if self.revin:
        #     output = self.revin_layer(output, 'denorm')

        return {
            "output": output,
            "encoder_attns": enc_out_dict['attns'],
        }
