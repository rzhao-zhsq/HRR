import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from models.layers.Autoformer_EncDec import series_decomp
from models.layers.PatchRevIN import RevIN
from utils.misc import neq_load_customized, freeze_params
from models.layers.Embed import DataEmbedding
from models.Conv import Conv1d, ConvStack, get_norm, get_activation
from models import Conv

# from models.layers.VAELayers import GaussianNet


def gaussian_kl_loss(posterior_mean, posterior_logvar, prior_mean, prior_logvar, norm_type="batchnorm"):
    kl_loss = -0.5 * torch.sum(
        1 + (posterior_logvar - prior_logvar)
        - torch.div(
            torch.pow(prior_mean - posterior_mean, 2) + posterior_logvar.exp(),
            prior_logvar.exp(),
        )
    )
    return kl_loss


def conditional_gaussian_kl_loss(posterior_mean, posterior_logvar, prior_mean, prior_logvar, norm_type="batchnorm"):
    kl_loss = -0.5 * torch.sum(
        1 + posterior_logvar - prior_logvar.exp() - prior_mean.pow(2)
        - torch.div(
            torch.pow(prior_mean - posterior_mean, 2) + posterior_logvar.exp(),
            prior_logvar.exp(),
        )
    )
    return kl_loss


class LN(nn.Module):
    def __init__(self, latent_dim, gamma=3.0):
        super().__init__()
        self.ln = nn.LayerNorm(latent_dim)
        self.ln.weight.requires_grad = False
        self.gamma = gamma
        self.reset_parameters()

    def reset_parameters(self):
        with torch.no_grad():
            self.ln.weight.fill_(self.gamma)

    def forward(self, x):
        return self.ln(x)


class StepWarmUpScheduler(object):
    def __init__(self, start_ratio, end_ratio, warmup_start_step, warmup_step):
        super().__init__()
        self.start_ratio = start_ratio
        self.end_ratio = end_ratio
        self.warmup_start_step = warmup_start_step
        self.warmup_step = warmup_step + int(warmup_step == 0)
        self.step_ratio = (end_ratio - start_ratio) / self.warmup_step
        # self.anneal_end = warmup_start + warmup_step
        # self.print_ratio_every = args.print_ratio_every

    def forward(self, step_num):
        if step_num < self.warmup_start_step:
            return self.start_ratio
        elif step_num >= self.warmup_step:
            return self.end_ratio
        else:
            ratio = self.start_ratio + self.step_ratio * (step_num - self.warmup_start_step)
            # if (step_num + 1) % self.print_ratio_every == 0:
            #     print("=" * 15, "STEP: {} RATIO:{}".format(step_num + 1, ratio), "=" * 15)
            return ratio


class GaussianNet(nn.Module):
    def __init__(
            self, c_in, c_out, z_dim, kernel=3, stride=1, gamma=0,
            temperature=1.0, valid_temperature=0.0,
            combine_type="vanilla",
            norm_pos=None,
            latent_norm=None,
            conv_gaussian=False,
    ):
        super(GaussianNet, self).__init__()

        self.temperature = temperature
        self.valid_temperature = valid_temperature
        self.combine_type = combine_type
        self.norm_pos = norm_pos
        self.z_dim = z_dim
        if conv_gaussian:
            self.posterior_mean = Conv1d(c_in, z_dim, kernel=kernel, stride=stride,)
            self.posterior_logvar = Conv1d(c_in, z_dim, kernel=kernel, stride=stride,)
            self.prior_mean = Conv1d(c_in, z_dim, kernel=kernel, stride=stride,)
            self.prior_logvar = Conv1d(c_in, z_dim, kernel=kernel, stride=stride,)
            if z_dim != c_out:
                self.recover_layer = Conv1d(z_dim, c_out, kernel=kernel, stride=stride,)
            else:
                self.recover_layer = nn.Identity()
        else:
            self.posterior_mean = nn.Linear(c_in, z_dim)
            self.posterior_logvar = nn.Linear(c_in, z_dim, bias=False)
            self.prior_mean = nn.Linear(c_in, z_dim, bias=False)
            self.prior_logvar = nn.Linear(c_in, z_dim, bias=False)
            self.recover_layer = nn.Linear(z_dim, c_out) if z_dim != c_out else nn.Identity()

        self.posteior_ln = LN(latent_dim=z_dim, gamma=gamma) if gamma > 0 else None
        self.prior_ln = LN(latent_dim=z_dim, gamma=gamma) if gamma > 0 else None
        self.ln = get_norm(latent_norm) if norm_pos is not None else nn.Identity()

        self.init_params(conv=conv_gaussian)

    def init_params(self, conv=False):
        if conv:
            nn.init.zeros_(self.prior_mean.model.weight)
            nn.init.zeros_(self.posterior_mean.model.weight)

            nn.init.normal_(self.prior_logvar.model.weight, std=0.01)
            nn.init.normal_(self.posterior_logvar.model.weight, std=0.01)
        else:
            nn.init.zeros_(self.prior_mean.weight)
            nn.init.zeros_(self.posterior_mean.weight)

            nn.init.normal_(self.prior_logvar.weight, std=0.01)
            nn.init.normal_(self.posterior_logvar.weight, std=0.01)

    def forward(self, prior, posterior, mixup=0.0, sample_size=1):
        # prior
        prior_mean = self.prior_mean(prior)
        prior_logvar = self.prior_logvar(prior)

        prior_z = GaussianNet.reparameterize(
            prior_mean, prior_logvar, is_logv=True,
            temperature=self.temperature if self.training else self.valid_temperature,
            sample_size=sample_size,
        )
        prior_output = self.combine(prior, prior_z, mixup, sample_size=sample_size,)

        # posterior
        # posterior_mean, posterior_logvar = self.posterior_net(posterior).chunk(2, dim=-1)
        posterior_mean = self.posterior_mean(posterior)
        posterior_logvar = self.posterior_logvar(posterior)
        posterior_z = GaussianNet.reparameterize(
            posterior_mean, posterior_logvar, is_logv=True,
            temperature=self.temperature if self.training else self.valid_temperature,
            sample_size=sample_size,
        )
        posterior_output = self.combine(posterior, posterior_z, mixup, sample_size=sample_size,)
        kl_loss = gaussian_kl_loss(
            posterior_mean=posterior_mean,
            posterior_logvar=posterior_logvar,
            prior_mean=prior_mean,
            prior_logvar=prior_logvar,
        )

        return {
            "prior": {
                "mean": prior_mean,
                "logvar": prior_logvar,
                "z": prior_z,
                "output": prior_output,
            },
            "posterior": {
                "mean": posterior_mean,
                "logvar": posterior_logvar,
                "z": posterior_z,
                "output": posterior_output,
            },
            "kl_loss": kl_loss
        }

    def combine(self, inputs, z, mixup=0.0, sample_size=1):
        if sample_size > 1:
            B, T, C = inputs.shape
            inputs = inputs.unsqueeze(0).expand(sample_size, -1, -1, -1).contiguous().view(-1, T, C)

        if self.norm_pos == "prefix":
            z = self.ln(z)

        # Tips: combine type: ["residual", "mixup", "gate_residual", "latent_only]
        if self.combine_type == "concatenate":
            inputs = torch.cat([inputs, z], dim=-1)
            outputs = self.concat_layer(inputs)
        elif self.combine_type == "gate_residual":
            z_recover = self.ln(self.recover_layer(z)) if self.norm_pos == "middle" else self.recover_layer(z)
            g = self.gate(torch.cat([inputs, z_recover], dim=-1))
            outputs = inputs * g + z_recover * (1 - g)
        elif self.combine_type == "mixup":
            assert 0. <= mixup <= 1.0
            z_recover = self.ln(self.recover_layer(z)) if self.norm_pos == "middle" else self.recover_layer(z)
            ind = torch.bernoulli(
                torch.full(inputs.size()[:2], mixup, device=inputs.device)
            ).int().unsqueeze(-1)
            outputs = inputs * (1 - ind) + z_recover * ind
        elif self.combine_type == "latent_only":
            z_recover = self.ln(self.recover_layer(z)) if self.norm_pos == "middle" else self.recover_layer(z)
            outputs = z_recover
        elif self.combine_type == "latent_residual":  # default residual
            z_recover = self.ln(self.recover_layer(z)) if self.norm_pos == "middle" else self.recover_layer(z)
            outputs = inputs + z_recover
        else:
            raise ValueError("Unknown combination type {}".format(self.combine_type))

        if self.norm_pos == "postfix":
            outputs = self.ln(outputs)
        return outputs

    @staticmethod
    def reparameterize(mean, var, is_logv=False, sample_size=1, temperature=1.0):
        if sample_size > 1:
            B, T, C = mean.shape
            mean = mean.unsqueeze(0).expand(sample_size, -1, -1, -1).contiguous().view(-1, T, C)
            var = var.unsqueeze(0).expand(sample_size, -1, -1, -1).contiguous().view(-1, T, C)

        if not is_logv:
            sigma = torch.sqrt(var + 1e-10)
        else:
            sigma = torch.exp(0.5 * var)

        epsilon = torch.randn_like(sigma)
        z = mean + epsilon * sigma * temperature
        return z


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.loginfo = self._set_loginfo_(configs)
        self.beta = configs.beta
        self.sample_size = configs.sample_size
        if configs.norm in ['layer']:
            norm_feature = configs.d_model
        elif configs.norm in ['instance', 'batch']:
            norm_feature = configs.pred_len
        else:
            norm_feature = None

        self.cnn = Conv.Model(configs)

        # Variational Module
        self.gaussian_net = GaussianNet(
            c_in=configs.d_model,
            c_out=configs.d_model,
            z_dim=configs.d_model,
            kernel=configs.kernel,
            stride=1,
            combine_type=configs.combine_type,
            norm_pos=configs.norm_pos,
            latent_norm=configs.latent_norm,
            conv_gaussian=configs.conv_gaussian,
            temperature=1.0,
            valid_temperature=configs.valid_temperature,
            gamma=0,
        )
        self.embedding = DataEmbedding(
            configs.enc_in+1, configs.d_model, configs.embed, configs.freq, configs.dropout
        )
        self.encoder = ConvStack(
            layers=configs.d_layers,
            c_in=configs.d_model,
            c_out=configs.d_model,
            kernel=configs.kernel,
            dropout=configs.dropout,
            activation=configs.activation,
            residual=configs.residual,
            norm=configs.norm,
            norm_feature=norm_feature,
        )
        self.decoder = ConvStack(
            layers=configs.d_layers,
            c_in=configs.d_model,
            c_out=configs.d_model,
            kernel=configs.kernel,
            dropout=configs.dropout,
            activation=configs.activation,
            residual=configs.residual,
            norm=configs.norm,
            norm_feature=norm_feature,
        )

        if configs.conv_head:
            self.regression_head = Conv1d(
                c_in=configs.d_model,
                c_out=configs.c_out,
                kernel=configs.kernel,
            )
        else:
            self.regression_head = nn.Linear(configs.d_model, configs.c_out, bias=True)

        if configs.load_ckpt:
            self.load_ckpt(configs.pretrained_model)

    @staticmethod
    def _set_loginfo_(cfg):

        return f"_s-{cfg.seq_len}_l-{cfg.label_len}_p-{cfg.pred_len}" \
               f"_d-{cfg.d_model}_el-{cfg.e_layers}_dl-{cfg.d_layers}" \
               f"_norm-{cfg.norm}" \
               f"_k-{cfg.kernel}" \
               f"_embed-{cfg.embed}" \
               f"-f-{cfg.freq}" \
               f"_drop-{cfg.dropout}" \
               f"{'-residual' if cfg.residual else ''}" \
               f"_beta-{cfg.beta}_{cfg.combine_type}_{cfg.norm_pos}_{cfg.latent_norm}"

    def get_loginfo(self):
        return self.loginfo

    def load_ckpt(self, pretrained_ckpt):
        """
        --load_ckpt
        --pretrained_model=/home2/rzhao/python/FEDformer/gef/Conv/CT/gefcom_reg_Conv_s-0_l-0_p-24_d-64_el-6_norm-none_embed-learned-f-h_ec-2_dc-2_oc-1_drop-0.3-residual/checkpoint.pth
        """
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
        neq_load_customized(self.cnn, checkpoint, verbose=True)

        # self.load_state_dict(load_dict)
        freeze_params(self.cnn)
        # freeze_params(self.encoder)
        # freeze_params(self.decoder)
        # freeze_params(self.regression_head)

    def generate(
        self,
        prev_x, prev_y, prev_mark,
        target_x, target_y, target_mark,
        **kwargs,
    ):
        cnn_out = self.cnn(prev_x, prev_y, prev_mark, target_x, target_y, target_mark, **kwargs)

        prior = self.embedding(torch.cat([target_x, cnn_out['output']], dim=-1), target_mark)
        posterior = self.embedding(torch.cat([target_x, target_y], dim=-1), target_mark)
        prior = self.encoder(prior)
        posterior = self.encoder(posterior)

        gaussian_out = self.gaussian_net.forward(prior=prior, posterior=posterior, sample_size=self.sample_size)

        prior = self.decoder(gaussian_out['prior']['output'])
        prior = self.regression_head(prior)
        B, T, C = target_y.shape
        output = prior.contiguous().view(self.sample_size, B, T, C).mean(0)

        return {
            "output": output,
            "kl_loss": gaussian_out['kl_loss'] / target_x.shape[1] * self.beta,
            # "y_hat_loss": F.mse_loss(cnn_out['output'], target_y),
            # "prior_loss": F.mse_loss(prior, target_y),
            "outputs": prior.contiguous().view(self.sample_size, B, T, C)
        }

    def forward(
            self,
            prev_x, prev_y, prev_mark,
            target_x, target_y, target_mark,
            **kwargs,
    ):
        if not self.training:
            return self.generate(
                prev_x, prev_y, prev_mark, target_x, target_y, target_mark, **kwargs,
            )
        # recognition net.
        cnn_out = self.cnn(prev_x, prev_y, prev_mark, target_x, target_y, target_mark, **kwargs)

        # encoder
        prior = self.embedding(torch.cat([target_x, cnn_out['output']], dim=-1), target_mark)
        posterior = self.embedding(torch.cat([target_x, target_y], dim=-1), target_mark)
        prior = self.encoder(prior)
        posterior = self.encoder(posterior)

        # gaussian net
        gaussian_out = self.gaussian_net.forward(prior=prior, posterior=posterior)

        # decoder
        posterior = self.decoder(gaussian_out['posterior']['output'])
        prior = self.decoder(gaussian_out['prior']['output'])

        # regression head
        posterior = self.regression_head(posterior)
        prior = self.regression_head(prior)

        output = posterior

        return {
            "output": output,
            "kl_loss": gaussian_out['kl_loss'] / target_x.shape[1] * self.beta,
            "y_hat_loss": F.mse_loss(cnn_out['output'], target_y),
            "prior_loss": F.mse_loss(prior, target_y),
            # "distill_loss": F.mse_loss(prior, posterior),
        }
