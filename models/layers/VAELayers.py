import torch
import torch.nn as nn


def reparameterize(mean, var, is_logv=False, sample_size=1):
    if sample_size > 1:
        mean = mean.contiguous().unsqueeze(1).expand(-1, sample_size, -1).reshape(-1, mean.size(-1))
        var = var.contiguous().unsqueeze(1).expand(-1, sample_size, -1).reshape(-1, var.size(-1))

    if not is_logv:
        sigma = torch.sqrt(var + 1e-10)
    else:
        sigma = torch.exp(0.5 * var)

    epsilon = torch.randn_like(sigma)
    z = mean + epsilon * sigma
    return z

def gaussian_kl_loss(posterior_mean, posterior_logvar, prior_mean, prior_logvar):
    kl_loss = -0.5 * torch.sum(
        1 + (posterior_logvar - prior_logvar)
        - torch.div(
            torch.pow(prior_mean - posterior_mean, 2) + posterior_logvar.exp(),
            prior_logvar.exp(),
        )
    )
    return kl_loss


class GaussianNet(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super().__init__()
        self.latent_dim = latent_dim
        self.mean = nn.Linear(input_dim, latent_dim)
        self.logvar = nn.Linear(input_dim, latent_dim)

    def forward(self, inputs, max_posterior=False, **kwargs):
        """
        :param inputs:  batch_size,input_dim
        :param max_posterior:
        :return:
            mean: batch_size, latent_dim
            logv: batch_size, latent_dim
            z: batch_size, latent_dim
            rec: batch_size, output_dim
        """
        mean, logvar, z = self.posterior(inputs, max_posterior=max_posterior)
        kl_loss = self.compute_kl_loss(mean=mean, logvar=logvar)

        return {"mean": mean, "logv": logvar, "z": z, "kl_loss": kl_loss}

    def posterior(self, inputs, max_posterior=False):
        mean = self.mean(inputs)
        logvar = self.logvar(inputs)
        z = reparameterize(mean, logvar, is_logv=True) if not max_posterior else mean
        return mean, logvar, z

    def prior(self, inputs, n=-1):
        if n < 0:
            n = inputs.size(0)
        z = torch.randn([n, self.latent_dim])

        if inputs is not None:
            z = z.to(inputs)

        return z

    @classmethod
    def compute_kl_loss(cls, mean, logvar):
        kl_loss = -0.5 * torch.sum(
            1 + logvar - torch.pow(mean, 2) - logvar.exp()
        )
        return kl_loss