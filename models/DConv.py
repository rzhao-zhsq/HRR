import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from models.layers.Autoformer_EncDec import series_decomp
from models.layers.PatchRevIN import RevIN
from utils.misc import neq_load_customized, freeze_params
from models.layers.VAELayers import GaussianNet

class Model(nn.Module):
    """
    Decomposition-Linear
    """
    def __init__(self, configs):
        super(Model, self).__init__()

        self.channels = configs.enc_in
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len

        # Decompsition Kernel Size
        # kernel_size = 25
        self.decompsition = series_decomp(configs.moving_avg)
        # self.individual = configs.individual if hasattr(configs, "individual") else True

        affine = configs.affine if hasattr(configs, "affine") else False
        subtract_last = configs.subtract_last if hasattr(configs, "subtract_last") else False
        self.revin = configs.revin if hasattr(configs, "revin") else True
        if self.revin:
            self.revin_layer = RevIN(configs.enc_in, affine=affine, subtract_last=subtract_last)

        encoder_kernel = 3
        decoder_kernel = 3
        self.encoder = nn.Sequential(
            nn.Conv1d(
                in_channels=configs.enc_in, out_channels=configs.d_model,
                kernel_size=(encoder_kernel,), padding=encoder_kernel//2, stride=(1,),
                padding_mode='circular', bias=False
            ),
            nn.ReLU(),
            nn.Dropout(configs.dropout),
            nn.AvgPool1d(kernel_size=3, stride=1, padding=1, ceil_mode=False, count_include_pad=False),
            # nn.MaxPool1d(kernel_size=3, padding=1, stride=1, ceil_mode=False,),
        )

        # TODO: variational auto-encoder pretraining.
        # self.gaussian_net = GaussianNet(input_dim=configs.seq_len, latent_dim=configs.seq_len)

        self.decoder = nn.Sequential(
            nn.Conv1d(
                in_channels=configs.d_model, out_channels=configs.c_out,
                kernel_size=(decoder_kernel,), padding=decoder_kernel//2,
                stride=(1,), padding_mode='circular', bias=False
            ),
            nn.ReLU(),
            nn.Dropout(configs.dropout),
            # nn.ConvTranspose1d(
            #     in_channels=configs.d_model, out_channels=configs.c_out,
            #     kernel_size=(decoder_kernel,), padding=(decoder_kernel // 2 -1,),
            #     stride=(2,), padding_mode='zeros', bias=False
            # ),
            # nn.GELU(),
            # nn.Dropout(configs.dropout),
            nn.AvgPool1d(kernel_size=3, stride=1, padding=1, ceil_mode=False, count_include_pad=False),
            # nn.MaxPool1d(kernel_size=decoder_kernel, padding=decoder_kernel//2, stride=1,  ceil_mode=False,),
        )

        self.predict_head = nn.Linear(configs.seq_len, configs.pred_len, bias=False)
        if configs.load_ckpt:
            self.load_ckpt(configs.pretrained_model)
            # freeze_params(self.decompsition)
            # freeze_params(self.revin_layer)
            # freeze_params(self.encoder)
            # freeze_params(self.gaussian_net)
            # freeze_params(self.decoder)

        self.loss_function = nn.MSELoss()

    def load_ckpt(self, pretrained_ckpt_path):
        # logger = get_logger()
        # logger.info('Load and Reinitialize Variational Network from pretrained ckpt {}'.format(pretrained_ckpt))
        checkpoint = torch.load(pretrained_ckpt_path, map_location='cpu')
        neq_load_customized(self, checkpoint, verbose=True)
        # self.load_state_dict(load_dict)

    def forward(self, batch_x, **kwargs):
        # x: [Batch, Input length, Channel]

        x = self.revin_layer(batch_x, 'norm') if self.revin else batch_x
        x, trend = self.decompsition(x)
        x, trend = x.transpose(1,2), trend.transpose(1, 2)

        x = self.encoder(x)

        # gaussin_net_output_dict = self.gaussian_net(x, max_posterior=True)
        # x = gaussin_net_output_dict['z']

        x = self.decoder(x)

        x = self.predict_head(x)
        trend = self.predict_head(trend)
        x = x + trend
        x = x.transpose(1,2)

        x = self.revin_layer(x, 'denorm') if self.revin else x

        mse_loss = self.loss_function(x, kwargs['target']) if kwargs.get("target", None) is not None else None
        return {
            "output": x,
            "batch_loss": mse_loss
        }
