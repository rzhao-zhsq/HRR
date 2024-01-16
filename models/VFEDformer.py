import torch
import torch.nn as nn
import torch.nn.functional as F
from models.layers.Embed import DataEmbedding_wo_pos
from models.layers.AutoCorrelation import AutoCorrelationLayer
from models.layers.FourierCorrelation import FourierBlock, FourierCrossAttention
from models.layers.MultiWaveletCorrelation import MultiWaveletCross, MultiWaveletTransform
from models.layers.Autoformer_EncDec import (
Encoder, Decoder, EncoderLayer, DecoderLayer, my_Layernorm, series_decomp, series_decomp_multi
)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def gaussian_kl_loss(posterior_mean, posterior_logvar, prior_mean, prior_logvar, norm_type="batchnorm"):
    kl_loss = -0.5 * torch.sum(
        1 + (posterior_logvar - prior_logvar)
        - torch.div(
            torch.pow(prior_mean - posterior_mean, 2) + posterior_logvar.exp(),
            prior_logvar.exp(),
        )
    )
    return kl_loss

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
                in_channels=configs.d_model, out_channels=configs.d_model,
                seq_len_q=self.seq_len // 2 + self.pred_len, seq_len_kv=self.seq_len,
                modes=configs.modes, ich=configs.d_model, base=configs.base, activation=configs.cross_activation,
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

        # vae part.
        v_attn = AutoCorrelationLayer(decoder_cross_att, configs.d_model, configs.n_heads)
        self.gaussian_net = GaussianNet(v_attn, configs.latent_dim, configs.d_model)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        # instance_norm
        # x_enc = self.instance_norm(x_enc, "norm")
        embedding = self.enc_embedding.forward(
            torch.cat([x_enc, x_dec[:, -self.pred_len:, :]], dim=1),
            torch.cat([x_mark_enc, x_mark_dec[:, -self.pred_len:, :]], dim=1)
        )

        # Tips: prior
        # enc
        enc_out, attns = self.encoder.forward(embedding[:, :x_enc.size(1), :])

        # Tips: posterior
        posterior_enc_out, posterior_enc_attns = self.encoder.forward(embedding[:, x_enc.size(1):, :])
        latent_out = self.gaussian_net(prior_inputs=enc_out, posterior_inputs=posterior_enc_out)

        enc_out = latent_out['posterior']['encoder_out'] if self.training else latent_out['prior']['encoder_out']

        # dec
        # decompose seasonal and trend
        mean = torch.mean(x_enc, dim=1).unsqueeze(1).repeat(1, self.pred_len, 1)
        zeros = torch.zeros([x_dec.shape[0], self.pred_len, x_dec.shape[2]]).to(device)  # cuda()

        seasonal_init, trend_init = self.decomp.forward(x_enc)
        trend_init = torch.cat([trend_init[:, -self.label_len:, :], mean], dim=1)
        seasonal_init = F.pad(seasonal_init[:, -self.label_len:, :], (0, 0, 0, self.pred_len))  # pad pred_len with zero
        # TODO(rzhao): decompose the seasonal_init with Variational Mode Decomposition (VMD).

        dec_embedding = self.dec_embedding.forward(seasonal_init, x_mark_dec)
        seasonal_part, trend_part = self.decoder.forward(
            dec_embedding, enc_out,
            x_mask=dec_self_mask, cross_mask=dec_enc_mask, trend=trend_init
        )
        # final
        dec_out = trend_part + seasonal_part
        # re_norm
        # dec_out = self.instance_norm(dec_out, "denorm")
        if x_mark_enc.size(-1) > x_enc.size(-1) and x_mark_enc.size(-1) % x_enc.size(-1) == 0:
            dec_out = dec_out.mean(-1, keepdim=True)
        kl_loss = gaussian_kl_loss(
            posterior_mean=latent_out['posterior']['mean'], posterior_logvar=latent_out['posterior']['logvar'],
            prior_mean=latent_out['prior']['mean'], prior_logvar=latent_out['prior']['mean'],
        )
        if self.output_attention:
            return dec_out[:, -self.pred_len:, :], attns, kl_loss
        return dec_out[:, -self.pred_len:, :],  kl_loss  # [B, L, D]


class GaussianNet(nn.Module):
    def __init__(self, attn, out_dim, latent_dim=8):
        super().__init__()
        input_dim = output_dim = out_dim
        latent_dim = latent_dim

        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.attention = attn
        self.posterior_net = nn.Linear(input_dim, latent_dim * 2)
        self.prior_net = nn.Sequential(
            nn.Linear(input_dim, latent_dim * 2 * 2),
            nn.Tanh(),  # non-linear
            nn.Linear(latent_dim * 2 * 2, latent_dim * 2),
        )
        self.recover_layer = nn.Sequential(
            nn.Linear(latent_dim, latent_dim * 2),
            nn.GELU(),
            nn.Linear(latent_dim * 2, output_dim),
            # nn.LayerNorm(output_dim),
        )

    def forward(self, prior_inputs, prior_mask=None, posterior_inputs=None, posterior_mask=None):
        temperature = 1.0 if self.training else 0.0  # for deterministic inference

        prior_attn = self.attention.forward(prior_inputs, prior_inputs, prior_inputs, prior_mask, self_attn=True)
        prior_mean, prior_logvar, prior_z = self.prior(prior_attn, temperature=temperature)
        prior_rec_z = self.recover_layer(prior_z if temperature > 0 else prior_mean)
        prior_gate_rec = self.combine(prior_inputs, prior_rec_z)
        if posterior_inputs is not None:
            posterior_attn = self.attention(posterior_inputs, prior_inputs, prior_inputs, posterior_mask)
            posterior_mean, posterior_logvar, posterior_z = self.posterior(posterior_attn)
            posterior_rec = self.recover_layer(posterior_z)
            posterior_gate_rec = self.combine(prior_inputs, posterior_rec)

        return {
            "posterior": {
                "mean": posterior_mean, "logvar": posterior_logvar, "z": posterior_z,
                "encoder_out": posterior_gate_rec,
            } if posterior_inputs is not None else None,
            "prior": {
                "mean": prior_mean, "logvar": prior_logvar, "z": prior_z,
                "encoder_out": prior_gate_rec,
            },
        }

    def posterior(self, inputs):
        mean, logvar = self.posterior_net(inputs).chunk(2, dim=-1)
        z = GaussianNet.reparameterize(mean, logvar, is_logv=True)
        return mean, logvar, z

    def prior(self, inputs, temperature=1.0):
        mean, logvar = self.prior_net(inputs).chunk(2, dim=-1)
        z = GaussianNet.reparameterize(mean, logvar, is_logv=True, temperature=temperature)
        return mean, logvar, z

    def combine(self, inputs, recoved_z):
        if self.gate is not None:
            g = self.gate(torch.cat([inputs, recoved_z], dim=-1))
            inputs = inputs * g + recoved_z * (1 - g)
        else:
            inputs = inputs + recoved_z
        return inputs


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
