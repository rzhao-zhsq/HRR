import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from models.layers.Autoformer_EncDec import series_decomp
from models.layers.PatchRevIN import RevIN
from models.layers.VAELayers import (
    reparameterize,gaussian_kl_loss,GaussianNet
)
from utils.misc import neq_load_customized


class Model(nn.Module):
    """
    Decomposition-Linear
    """
    def __init__(self, configs):
        super(Model, self).__init__()
        if configs.ssl:
            configs.pred_len = configs.seq_len

        self.channels = configs.enc_in
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.ssl = configs.ssl

        # Decompsition Kernel Size
        # kernel_size = 25
        self.decompsition = series_decomp(configs.moving_avg)
        # self.individual = configs.individual if hasattr(configs, "individual") else True

        affine = configs.affine if hasattr(configs, "affine") else False
        subtract_last = configs.subtract_last if hasattr(configs, "subtract_last") else False
        self.revin = configs.revin if hasattr(configs, "revin") else True
        if self.revin:
            self.revin_layer = RevIN(configs.enc_in, affine=affine, subtract_last=subtract_last)

        self.encoder = nn.Sequential(
            nn.Conv1d(
                in_channels=configs.enc_in, out_channels=configs.d_model,
                kernel_size=(3,), padding=1, stride=(1,), padding_mode='circular', bias=False
            ),
            nn.Dropout(configs.dropout),
            nn.BatchNorm1d(num_features=configs.d_model),
            # nn.AvgPool1d(
            #     kernel_size=3, stride=1, padding=1, ceil_mode=False, count_include_pad=True
            # ),

            # nn.Conv1d(
            #     in_channels=configs.d_model, out_channels=configs.d_model * 2,
            #     kernel_size=(3,), padding=1, stride=(1,), padding_mode='circular', bias=False
            # ),
            # nn.BatchNorm1d(num_features=configs.d_model * 2),
            # nn.Dropout(configs.dropout),
        )

        # TODO: variational auto-encoder pretraining.
        self.gaussian_net = GaussianNet(input_dim=configs.seq_len, latent_dim=configs.seq_len)

        self.decoder = nn.Sequential(
            nn.Conv1d(
                in_channels=configs.d_model, out_channels=configs.c_out,
                kernel_size=(3,), padding=1, stride=(1,), padding_mode='circular', bias=False
            ),
            nn.Dropout(configs.dropout),
            nn.BatchNorm1d(num_features=configs.c_out),
            # nn.AvgPool1d(
            #     kernel_size=3, stride=1, padding=1, ceil_mode=False, count_include_pad=True
            # ),

            # nn.Conv1d(
            #     in_channels=configs.d_model, out_channels=configs.c_out,
            #     kernel_size=(3,), padding=1, stride=(1,), padding_mode='circular', bias=False
            # ),
            # nn.BatchNorm1d(num_features=configs.c_out),
            # nn.Dropout(configs.dropout),

        )
        self.predict_head = nn.Linear(configs.seq_len, configs.pred_len, bias=True)
        if not configs.ssl:
            self.load_ckpt(configs.pretrained_model)

        self.mse_loss_function = nn.MSELoss()


    def load_ckpt(self, pretrained_ckpt):
        # logger = get_logger()
        # logger.info('Load and Reinitialize Variational Network from pretrained ckpt {}'.format(pretrained_ckpt))
        checkpoint = torch.load(pretrained_ckpt, map_location='cpu')
        # load_dict = {}
        # for k, v in checkpoint.items():
        #     if "mlm" not in k:
        #         load_dict[k] = v
        #         # if "translation_network" in k:
        #         #     load_dict[k.replace('translation_network.', '')] = v
        #         # else:
        #     else:
        #         logger.info('{} not loaded.'.format(k))
        # # load_dict.pop()
        neq_load_customized(self, checkpoint['model_state'], verbose=True)
        # self.load_state_dict(load_dict)

    def forward(self, batch_x, **kwargs):
        raw_x = batch_x
        x = self.revin_layer(batch_x, 'norm') if self.revin else batch_x
        x, trend = self.decompsition(x)
        x = self.encoder(x.transpose(1,2))
        gaussin_net_output_dict = self.gaussian_net(x)
        x = self.decoder(gaussin_net_output_dict['z'])
        x = self.predict_head(x)
        trend = self.predict_head(trend.transpose(1, 2))
        x = x + trend
        x = x.transpose(1,2)

        x = self.revin_layer(x, 'denorm') if self.revin else x

        # compute loss
        reconstruction_loss = self.mse_loss_function(raw_x, x)
        kl_loss = gaussin_net_output_dict['kl_loss'] / x.size(0)
        total_loss = reconstruction_loss + kl_loss

        return {
            "output": x,
            "batch_loss": total_loss,
            "reconstruction_loss": reconstruction_loss,
            "kl_loss": kl_loss,
        }
