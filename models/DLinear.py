import torch.nn as nn

from models.layers.Autoformer_EncDec import series_decomp


class Model(nn.Module):
    """
    Decomposition-Linear
    """
    def __init__(self, configs):
        super(Model, self).__init__()
        if configs.label_len != 0:
            print(f"For DLinear, label length must be 0, automatically set from {configs.label_len} to 0 now!")
            configs.label_len = 0
        self.channels = configs.enc_in
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len

        # Decompsition Kernel Size
        # kernel_size = 25
        self.moving_avg = configs.moving_avg
        self.decompsition = series_decomp(configs.moving_avg)
        self.individual = configs.individual =configs.individual if hasattr(configs, "individual") else True

        if self.individual:
            self.Linear_Seasonal = nn.Linear(configs.seq_len, configs.enc_in * configs.pred_len)
            self.Linear_Trend = nn.Linear(configs.seq_len, configs.enc_in * configs.pred_len)

        else:
            self.Linear_Seasonal = nn.Linear(self.seq_len, self.pred_len)
            self.Linear_Trend = nn.Linear(self.seq_len, self.pred_len)
            
            # Use this two lines if you want to visualize the weights
            # self.Linear_Seasonal.weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len,self.seq_len]))
            # self.Linear_Trend.weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len,self.seq_len]))

        self.loginfo = self._set_loginfo_(configs)

    def forward(self, batch_x, **kwargs):
        # x: [Batch, Input length, Channel]
        seasonal_init, trend_init = self.decompsition(batch_x)

        seasonal_output = self.Linear_Seasonal(seasonal_init.permute(0, 2, 1))
        trend_output = self.Linear_Trend(trend_init.permute(0, 2, 1))
        x = seasonal_output + trend_output

        if self.individual:
            x = x.reshape(-1, self.channels, self.pred_len, self.channels).permute(1, 3, 0, 2).diagonal()
        else:
            x = x.permute(0, 2, 1)  # to [Batch, Output length, Channel]
        return {
            "output": x
        }

    def _set_loginfo_(self, cfg):
        return f"_s-{cfg.seq_len}_l-{cfg.label_len}_p-{cfg.pred_len}" \
               f"-freq_{cfg.freq}_encin_{cfg.enc_in}_kernel_{cfg.moving_avg}_individual_{cfg.individual}"

    def get_loginfo(self):

        return self.loginfo
