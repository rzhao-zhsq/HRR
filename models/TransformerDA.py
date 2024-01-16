import torch
import torch.nn as nn

from models.layers.Attention import ResembleFullAttention, AttentionLayer
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
        self.diversity_weight = configs.diversity_weight
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
                        ResembleFullAttention(
                            False, configs.factor, attention_dropout=configs.dropout,
                            output_attention=configs.output_attention,
                            sub_space=configs.sub_space, sub_space_residual=configs.sub_space_residual,
                            out_space=configs.out_space, out_space_residual=configs.out_space_residual,
                            diversity_metric=configs.diversity_metric,
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
        self.decoder = Decoder(
            [
                DecoderLayer(
                    # self-attn
                    AttentionLayer(
                        ResembleFullAttention(
                            True, configs.factor, attention_dropout=configs.dropout, output_attention=False,
                            sub_space=configs.sub_space, sub_space_residual=configs.sub_space_residual,
                            out_space=configs.out_space, out_space_residual=configs.out_space_residual,
                            diversity_metric=configs.diversity_metric,
                        ),
                        configs.d_model,
                        configs.n_heads
                    ),
                    # cross-attn
                    AttentionLayer(
                        ResembleFullAttention(
                            False, configs.factor, attention_dropout=configs.dropout, output_attention=False,
                            sub_space=configs.sub_space, sub_space_residual=configs.sub_space_residual,
                            out_space=configs.out_space, out_space_residual=configs.out_space_residual,
                            diversity_metric=configs.diversity_metric,
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
            projection=nn.Linear(configs.d_model, configs.c_out, bias=True)
        )
        self.loginfo = self._set_loginfo_(configs)

    def _set_loginfo_(self, cfg):

        return f"_sl_{cfg.seq_len}_ll_{cfg.label_len}_pl_{cfg.pred_len}" \
               f"-freq_{cfg.freq}" \
               f"_divmetric_{cfg.diversity_metric}_divweight_{cfg.diversity_weight}" \
               f"_sub_{cfg.sub_space}_subres_{cfg.sub_space_residual}" \
               f"_out_{cfg.out_space}_outres_{cfg.out_space_residual}" \
               f"_embed_{cfg.embed}_encin_{cfg.enc_in}_decin_{cfg.dec_in}_deout_{cfg.c_out}" \
               f"_el_{cfg.e_layers}_dl_{cfg.d_layers}_dmodel_{cfg.d_model}_nhead_{cfg.n_heads}_dff_{cfg.d_ff}" \
               f"_dropout_{cfg.dropout}"

    def get_loginfo(self):
        return self.loginfo

    def forward(
            self,
            batch_x,
            batch_x_mark,
            dec_inp,
            batch_y_mark,
            # x_enc, x_mark_enc, x_dec, x_mark_dec,
            enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None,
            **kwargs,
    ):

        enc_out = self.enc_embedding(batch_x, batch_x_mark)
        enc_out_dict = self.encoder(enc_out, attn_mask=enc_self_mask)

        dec_out = self.dec_embedding(dec_inp, batch_y_mark)
        dec_out_dict = self.decoder(dec_out, enc_out_dict['output'], x_mask=dec_self_mask, cross_mask=dec_enc_mask)

        output = dec_out_dict['output'][:, -self.pred_len:, :]

        enc_sub_dis_loss = enc_out_dis_loss = torch.tensor(0.).to(output.device)
        dec_sub_dis_loss = dec_out_dis_loss = torch.tensor(0.).to(output.device)

        for disagreement in enc_out_dict['sub_disagreements']:
            enc_sub_dis_loss += disagreement.mean() if disagreement is not None else 0
        for disagreement in enc_out_dict['out_disagreements']:
            enc_out_dis_loss += disagreement.mean() if disagreement is not None else 0
        for disagreement in dec_out_dict['sub_disagreements']:
            dec_sub_dis_loss += disagreement.mean() if disagreement is not None else 0
        for disagreement in dec_out_dict['out_disagreements']:
            dec_out_dis_loss += disagreement.mean() if disagreement is not None else 0

        disagreement_loss = (
                enc_sub_dis_loss
                + enc_out_dis_loss
                + dec_sub_dis_loss
                + dec_out_dis_loss
        ) * self.diversity_weight

        return {
            # pred output
            "output": output,
            # disagreement_loss
            "disagreement_loss": disagreement_loss,
            # # encoder output
            # "encoder_output": enc_out_dict['output'],
            # "encoder_attns": enc_out_dict['attns'],
            # "encoder_sub_disagreements": enc_out_dict['sub_disagreements'],
            # "encoder_out_disagreements": enc_out_dict['out_disagreements'],
            # # decoder output
            # "decoder_output": dec_out_dict['output'],
            # "decoder_attns": dec_out_dict['attns'],
            # "decoder_sub_disagreements": dec_out_dict['sub_disagreements'],
            # "decoder_out_disagreements": dec_out_dict['out_disagreements'],
            # "decoder_cross_attns": dec_out_dict['cross_attns'],
            # "decoder_cross_sub_disagreements": dec_out_dict['cross_sub_disagreements'],
            # "decoder_cross_out_disagreements": dec_out_dict['cross_out_disagreements'],
        }

