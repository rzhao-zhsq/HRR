import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from models.layers.Autoformer_EncDec import series_decomp
from models.layers.Embed import DataEmbedding
from models.layers.PatchRevIN import RevIN
from models.layers.VAELayers import (
reparameterize,gaussian_kl_loss,GaussianNet
)


class Model(nn.Module):
    """
    Decomposition-Linear
    """
    def __init__(self, configs):
        super(Model, self).__init__()
        self.channels = configs.enc_in
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.ssl = configs.ssl

        affine = configs.affine if hasattr(configs, "affine") else False
        subtract_last = configs.subtract_last if hasattr(configs, "subtract_last") else False
        self.revin = configs.revin if hasattr(configs, "revin") else True
        if self.revin:
            self.revin_layer = RevIN(configs.enc_in, affine=affine, subtract_last=subtract_last)

        # Decompsition Kernel Size
        # kernel_size = 25
        # self.decompsition = series_decomp(configs.moving_avg)
        self.individual = configs.individual if hasattr(configs, "individual") else True
        # self.embedding = DataEmbedding(
        #     configs.enc_in, configs.d_model, configs.embed, configs.freq, configs.dropout
        # )
        # self.decompsition = series_decomp(configs.moving_avg)
        # self.encoder = nn.Sequential(
        #     nn.Linear(configs.seq_len, configs.d_model),
        #     nn.Dropout(configs.dropout),
        #     nn.Tanh(),
        # )
        self.gaussian_net = GaussianNet(input_dim=configs.seq_len, latent_dim=configs.d_model)
        # self.decoder = nn.Linear(configs.d_model, configs.pred_len)
        self.decoder = nn.Sequential(
            nn.Linear(configs.d_model, configs.pred_len),
            # # nn.Dropout(configs.dropout),
            # nn.ReLU(),
            # nn.Linear(configs.d_model, configs.pred_len),
            # nn.Tanh(),
        )
        self.loss_function = nn.MSELoss()

    def forward(self, x, **kwargs):
        raw_x = x
        # x: [Batch, Input length, Channel]

        x = self.revin_layer(x, 'norm') if self.revin else x

        # x = self.embedding(x)

        # seasonal_init, trend_init = self.decompsition(x)
        # x = self.encoder(x.transpose(1,2))
        gaussin_net_output_dict = self.gaussian_net(x.transpose(1,2), max_posterior=not self.training)
        kl_loss = gaussin_net_output_dict['kl_loss'] / x.size(0)
        x = self.decoder(gaussin_net_output_dict['z']).transpose(1,2)

        x = self.revin_layer(x, 'denorm') if self.revin else x

        return {
            "output": x,
            "reconstruction_loss": self.ssl_loss(raw_x, x) if self.ssl else None,
            "kl_loss": kl_loss
        }

