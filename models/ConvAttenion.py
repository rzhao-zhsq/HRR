import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from models.layers.Autoformer_EncDec import series_decomp
from models.layers.PatchRevIN import RevIN
from utils.misc import neq_load_customized, freeze_params
from models.layers.Embed import DataEmbedding
from models.layers.VAELayers import GaussianNet


def get_norm(norm_type: str = "layer_norm", num_feature: int = None):
    if norm_type:
        assert isinstance(norm_type, str), "Unknown activation type {}".format(norm_type)
        if norm_type.lower() == "layer":
            return nn.LayerNorm(num_feature)
        elif norm_type.lower() == "instance":
            return nn.InstanceNorm1d(num_feature)
        elif norm_type.lower() == "batch":
            return nn.BatchNorm1d(num_feature)
        else:
            return nn.Identity()
    else:
        return nn.Identity()

def get_activation(activation):
    if activation.lower() == "relu":
        return nn.ReLU()
    elif activation.lower() == "relu6":
        return nn.ReLU6()
    elif activation.lower() == "prelu":
        return nn.PReLU()
    elif activation.lower() == "selu":
        return nn.SELU()
    elif activation.lower() == "celu":
        return nn.CELU()
    elif activation.lower() == "gelu":
        return nn.GELU()
    elif activation.lower() == "sigmoid":
        return nn.Sigmoid()
    elif activation.lower() == "softplus":
        return nn.Softplus()
    elif activation.lower() == "softshrink":
        return nn.Softshrink()
    elif activation.lower() == "softsign":
        return nn.Softsign()
    elif activation.lower() == "tanh":
        return nn.Tanh()
    elif activation.lower() == "tanhshrink":
        return nn.Tanhshrink()
    else:
        raise ValueError("Unknown activation type {}".format(activation))


class Conv1d(nn.Module):
    def __init__(self, c_in, c_out, kernel=3, stride=1,):
        super(Conv1d, self).__init__()
        self.model = nn.Conv1d(
            in_channels=c_in, out_channels=c_out,
            kernel_size=(kernel,), padding=kernel//2,
            stride=(stride,), padding_mode='circular', bias=False
        )
        for module in self.modules():
            if isinstance(module, nn.Conv1d):
                nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        return self.model(x.transpose(1, 2)).transpose(1, 2)


class ConvLayer(nn.Module):
    def __init__(
            self,
            c_in, c_out, kernel=3, stride=1,
            dropout=0.1,
            activation="tanh",
            residual=False, norm="", norm_feature=None,
    ):
        super(ConvLayer, self).__init__()
        self.residual = residual
        self.conv = Conv1d(c_in=c_in, c_out=c_out, kernel=kernel, stride=stride,)
        self.activation = get_activation(activation)
        self.norm = get_norm(norm, norm_feature)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        residual = x if self.residual else 0.
        x = self.conv(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.norm(x + residual)
        return x


class ConvStack(nn.Module):
    def __init__(
            self,
            c_in, c_out, kernel=3, stride=1, layers=1,
            dropout=0.1,
            activation="tanh",
            residual=False, norm="", norm_feature=None,
    ):
        super(ConvStack, self).__init__()
        self.residual = residual
        self.layers = nn.ModuleList(
            [
                ConvLayer(
                    c_in, c_out, kernel=kernel, stride=stride,
                    dropout=dropout,
                    activation=activation,
                    residual=residual, norm=norm, norm_feature=norm_feature,
                ) for _ in range(layers)
            ]
        )

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()

        # self.channels = configs.enc_in
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.loginfo = self._set_loginfo_(configs)

        if configs.norm in ['layer']:
            norm_feature = configs.d_model
        elif configs.norm in ['instance', 'batch']:
            norm_feature = configs.pred_len
        else:
            norm_feature = None
        self.embedding = DataEmbedding(
            configs.enc_in, configs.d_model, configs.embed, configs.freq, configs.dropout
        )
        self.cnn = ConvStack(
            c_in=configs.d_model,
            c_out=configs.d_model,
            kernel=configs.kernel,
            layers=configs.e_layers,
            dropout=configs.dropout,
            activation=configs.activation,
            residual=configs.residual,
            norm=configs.norm,
            norm_feature=norm_feature,
        )
        if configs.conv_head:
            self.regression_head = Conv1d(configs.d_model, configs.c_out)
        else:
            self.regression_head = nn.Linear(configs.d_model, configs.c_out, bias=True)

    @staticmethod
    def _set_loginfo_(cfg):

        return f"_s-{cfg.seq_len}_l-{cfg.label_len}_p-{cfg.pred_len}" \
               f"_d-{cfg.d_model}_el-{cfg.e_layers}" \
               f"_norm-{cfg.norm}" \
               f"_k-{cfg.kernel}" \
               f"_embed-{cfg.embed}" \
               f"-f-{cfg.freq}" \
               f"_drop-{cfg.dropout}" \
               f"{'-residual' if cfg.residual else ''}"

    def get_loginfo(self):
        return self.loginfo

    def forward(
            self,
            prev_x, prev_y, prev_mark,
            target_x, target_y, target_mark,
            **kwargs,
    ):
        x = self.embedding(target_x, target_mark)

        x = self.cnn(x)

        x = self.regression_head(x)

        return {
            "output": x,
            # "batch_loss": mse_loss
        }