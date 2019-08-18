import torch
import torch.nn as nn
import torch.distributions as distrib
import torch.distributions.transforms as transform

from .. import NormalizingFlow
from ..flows import CoupledLayerFlow, BatchNormFlow

class RealNVP(nn.Module):
    def __init__(self, n_blocks, input_size, hidden_size, n_hidden, base_dist, batch_norm=True):
        super().__init__()

        # base distribution for calculation of log prob under the model
        self.base_dist = base_dist

        # construct model
        modules = []
        mask = torch.arange(input_size).float() % 2
        for i in range(n_blocks):
            modules += [CoupledLayerFlow(input_size, hidden_size, n_hidden, mask, cond_label_size)]
            mask = 1 - mask
            modules += batch_norm * [BatchNormFlow(input_size)]

        self.net = NormalizingFlow(*modules, base_dist=base_dist)

    def forward(self, z):
        return self.net(z)

    def inverse(self, x):
        return self.net.inverse(x)

    def log_prob(self, x):
        z, log_abs_det_jacobian = self.inverse(x)

        base_log_probs = self.base_dist.log_prob(z)

        if len(base_log_probs.shape) > 1:
            return base_log_probs.sum(dim=1) + log_abs_det_jacobian
        else:
            return base_log_probs + log_abs_det_jacobian

    def sample(self, n_samples):
        with torch.no_grad():
            z_samples = self.base_dist.sample((n_samples, ))
            return self.forward(z_samples)[0]

    def to(self, device="cuda:0"):
        super(RealNVP, self).to(device)

        self.net.to(device)

        for key in self.base_density.__dict__:
            if hasattr(getattr(self.base_density, key), "to"):
                setattr(self.base_dist, key, getattr(self.base_dist, key).to(device))
