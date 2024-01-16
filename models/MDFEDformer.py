import torch
import torch.nn as nn
import torch.nn.functional as F
from models.layers.Embed import DataEmbedding, DataEmbedding_wo_pos
from models.layers.AutoCorrelation import AutoCorrelation, AutoCorrelationLayer
from models.layers.FourierCorrelation import FourierBlock, FourierCrossAttention
from models.layers.MultiWaveletCorrelation import MultiWaveletCross, MultiWaveletTransform
from models.layers.Attention import FullAttention, ProbAttention
from models.layers.Autoformer_EncDec import (
    Encoder, Decoder, EncoderLayer, DecoderLayer, my_Layernorm, series_decomp, series_decomp_multi
)
import math
import numpy as np
from models.layers.RevIN import RevIN
from PyEMD import EMD


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Model(nn.Module):
    """
    FEDformer performs the attention mechanism on frequency domain and achieved O(N) complexity
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.version = configs.version
        self.mode_select = configs.mode_select
        self.modes = configs.modes
        self.seq_len = configs.seq_len
        self.label_len = configs.label_len
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention
        # self.instance_norm = RevIN(configs.enc_in)

        # Decomp
        kernel_size = configs.moving_avg
        if isinstance(kernel_size, list):
            self.decomp = series_decomp_multi(kernel_size)
        else:
            self.decomp = series_decomp(kernel_size)
        self.emd = EMD()

        # Embedding
        # The series-wise connection inherently contains the sequential information.
        # Thus, we can discard the position embedding of transformers.
        self.enc_embedding = DataEmbedding_wo_pos(
            configs.enc_in, configs.d_model, configs.embed,
            configs.freq, configs.dropout
        )
        self.dec_embedding = DataEmbedding_wo_pos(
            configs.dec_in, configs.d_model, configs.embed,
            configs.freq, configs.dropout
        )

        if configs.version == 'Wavelets':
            encoder_self_att = MultiWaveletTransform(ich=configs.d_model, L=configs.L, base=configs.base)
            decoder_self_att = MultiWaveletTransform(ich=configs.d_model, L=configs.L, base=configs.base)
            decoder_cross_att = MultiWaveletCross(
                in_channels=configs.d_model,
                out_channels=configs.d_model,
                seq_len_q=self.seq_len // 2 + self.pred_len,
                seq_len_kv=self.seq_len,
                modes=configs.modes,
                ich=configs.d_model,
                base=configs.base,
                activation=configs.cross_activation,
            )
        else:
            encoder_self_att = FourierBlock(
                in_channels=configs.d_model, out_channels=configs.d_model, seq_len=self.seq_len,
                modes=configs.modes, mode_select_method=configs.mode_select
            )
            decoder_self_att = FourierBlock(
                in_channels=configs.d_model, out_channels=configs.d_model, seq_len=self.seq_len // 2 + self.pred_len,
                modes=configs.modes, mode_select_method=configs.mode_select
            )
            decoder_cross_att = FourierCrossAttention(
                in_channels=configs.d_model, out_channels=configs.d_model, seq_len_q=self.seq_len // 2 + self.pred_len,
                seq_len_kv=self.seq_len, modes=configs.modes, mode_select_method=configs.mode_select
            )
        # Encoder
        enc_modes = int(min(configs.modes, configs.seq_len // 2))
        dec_modes = int(min(configs.modes, (configs.seq_len // 2 + configs.pred_len) // 2))
        print('enc_modes: {}, dec_modes: {}'.format(enc_modes, dec_modes))

        self.encoder = Encoder(
            [
                EncoderLayer(
                    AutoCorrelationLayer(encoder_self_att, configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    moving_avg=configs.moving_avg,
                    dropout=configs.dropout,
                    activation=configs.activation,
                ) for _ in range(configs.e_layers)
            ],
            norm_layer=my_Layernorm(configs.d_model)
        )
        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AutoCorrelationLayer(decoder_self_att, configs.d_model, configs.n_heads),
                    AutoCorrelationLayer(decoder_cross_att, configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.c_out,
                    configs.d_ff,
                    moving_avg=configs.moving_avg,
                    dropout=configs.dropout,
                    activation=configs.activation,
                ) for _ in range(configs.d_layers)
            ],
            norm_layer=my_Layernorm(configs.d_model),
            projection=nn.Linear(configs.d_model, configs.c_out, bias=True)
        )

    def emd_decompose(self, x: torch.Tensor, max_imf=3):
        x_numpy = x.detach().cpu().numpy()
        B, L, T = x.shape
        imfs = np.zeros([B, L, T, max_imf], dtype=np.float32)
        trends = np.zeros([B, L, T], dtype=np.float32)
        for i in range(B):
            for j in range(T):
                self.emd.emd(x_numpy[i, :, j], max_imf=max_imf)
                imf, trend = self.emd.get_imfs_and_trend()
                if imf.shape[0] == max_imf:
                    imfs[i, :, j, :], trends[i, :, j] = imf.T, trend
                else:
                    for k in range(imf.shape[0]):
                        imfs[i, :, j, k] = imf.T[:, k]
        return torch.from_numpy(imfs).to(x.device), torch.from_numpy(trends).to(x.device)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        # instance_norm
        # x_enc = self.instance_norm(x_enc, "norm")

        # enc
        enc_out = self.enc_embedding.forward(x_enc, x_mark_enc)  # [B, seq_len, 512]
        enc_out, attns = self.encoder.forward(enc_out, attn_mask=enc_self_mask)

        # dec
        # decompose seasonal and trend
        mean = torch.mean(x_enc, dim=1).unsqueeze(1).repeat(1, self.pred_len, 1)
        zeros = torch.zeros([x_dec.shape[0], self.pred_len, x_dec.shape[2]]).to(device)  # cuda()

        # seasonal_init, trend_init = self.decomp.forward(x_enc)
        seasonal_inits, trend_init = self.emd_decompose(x_enc, max_imf=3)
        trend_init = torch.cat([trend_init[:, -self.label_len:, :], mean], dim=1)

        dec_out = torch.zeros_like(x_dec, device=x_dec.device)
        for i in range(3):
            seasonal_init = seasonal_inits[..., i]
            seasonal_init = F.pad(seasonal_init[:, -self.label_len:, :], (0, 0, 0, self.pred_len))

            dec_embedding = self.dec_embedding.forward(seasonal_init, x_mark_dec)
            seasonal_part, trend_part = self.decoder.forward(
                dec_embedding, enc_out,
                x_mask=dec_self_mask, cross_mask=dec_enc_mask, trend=trend_init
            )

            # final
            dec_out += trend_part + seasonal_part
        # re_norm

        # dec_out = self.instance_norm(dec_out, "denorm")

        if x_mark_enc.size(-1) > x_enc.size(-1) and x_mark_enc.size(-1) % x_enc.size(-1) == 0:
            dec_out = dec_out.mean(-1, keepdim=True)
        if self.output_attention:
            return dec_out[:, -self.pred_len:, :], attns
        return dec_out[:, -self.pred_len:, :]  # [B, L, D]


if __name__ == '__main__':
    class Configs(object):
        ab = 0
        modes = 32
        mode_select = 'random'
        # version = 'Fourier'
        version = 'Wavelets'
        moving_avg = [12, 24]
        L = 1
        base = 'legendre'
        cross_activation = 'tanh'
        seq_len = 96
        label_len = 48
        pred_len = 96
        output_attention = True
        enc_in = 7
        dec_in = 7
        d_model = 16
        embed = 'timeF'
        dropout = 0.05
        freq = 'h'
        factor = 1
        n_heads = 8
        d_ff = 16
        e_layers = 2
        d_layers = 1
        c_out = 7
        activation = 'gelu'
        wavelet = 0


    configs = Configs()
    model = Model(configs)

    print('parameter number is {}'.format(sum(p.numel() for p in model.parameters())))
    enc = torch.randn([3, configs.seq_len, 7])
    enc_mark = torch.randn([3, configs.seq_len, 4])

    dec = torch.randn([3, configs.seq_len // 2 + configs.pred_len, 7])
    dec_mark = torch.randn([3, configs.seq_len // 2 + configs.pred_len, 4])
    out = model.forward(enc, enc_mark, dec, dec_mark)
    print(out)
