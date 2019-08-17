import torch
import torch.nn as nn
import torch.distributions as distrib
import torch.distributions.transforms as transform

from .flow import Flow

class BatchNormFlow(Flow):
    def __init__(self, dim, eps=1e-5, momentum=0.9):
        super(BatchNormFlow, self).__init__()
        self.bijective = True

        self.momentum = momentum
        self.eps = eps

        self.log_gamma = nn.Parameter(torch.zeros(dim))
        self.beta = nn.Parameter(torch.zeros(dim))

        self.register_buffer('running_mean', torch.zeros(dim))
        self.register_buffer('running_var', torch.ones(dim))

    def _call(self, z):
        if self.training:
            self.batch_mean = z.mean(0)
            self.batch_var = z.var(0) # note MAF paper uses biased variance estimate; ie x.var(0, unbiased=False)

            # update running mean
            self.running_mean.mul_(self.momentum).add_(self.batch_mean.data * (1 - self.momentum))
            self.running_var.mul_(self.momentum).add_(self.batch_var.data * (1 - self.momentum))

            mean = self.batch_mean
            var = self.batch_var
        else:
            mean = self.running_mean
            var = self.running_var

        # compute normalized input (cf original batch norm paper algo 1)
        z_hat = (z - mean) / torch.sqrt(var + self.eps)
        x = self.log_gamma.exp() * z_hat + self.beta

        return x

    def _inverse(self, x):
        if self.training:
            mean = self.batch_mean
            var = self.batch_var
        else:
            mean = self.running_mean
            var = self.running_var

        z_hat = (x - self.beta) * torch.exp(-self.log_gamma)
        z = z_hat * torch.sqrt(var + self.eps) + mean
        return z

    def log_abs_det_jacobian(self, z, x):
        var = (self.batch_var if self.training else self.running_var)

        log_abs_det_jacobian = 0.5 * torch.log(var + self.eps) - self.log_gamma

        return (log_abs_det_jacobian.sum()
                    .repeat(z.size(0), 1)
                    .squeeze())

    def to(self, device="cuda:0"):
        super(BatchNormFlow, self).to(device)
