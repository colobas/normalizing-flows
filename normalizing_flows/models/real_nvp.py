import torch
import torch.nn as nn
import torch.distributions as distrib
import torch.distributions.transforms as transform

from .. import NormalizingFlow
from ..flows import CouplingLayerFlow, BatchNormFlow

class RealNVP(nn.Module):
    def __init__(self, n_blocks, xdim, hdim, n_hidden, base_dist, batch_norm=True):
        super().__init__()

        # base distribution for calculation of log prob under the model
        self.base_dist = base_dist

        # construct model
        modules = []
        mask = torch.arange(xdim) % 2
        for i in range(n_blocks):
            modules += [CouplingLayerFlow(xdim, hdim, n_hidden, mask)]
            mask = 1 - mask
            if batch_norm:
                modules += [BatchNormFlow(xdim)]

        self.net = NormalizingFlow(*modules, base_dist=base_dist)

    def forward(self, z):
        return self.net(z)

    def inverse(self, x):
        return self.net.inverse(x)

    def log_prob(self, x):
        return self.net.log_prob(x)

    def sample(self, n_samples):
        return self.net.sample(n_samples)

    def to(self, device):
        super(RealNVP, self).to(device)

        self.net.to(device)

        for key in self.base_dist.__dict__:
            if hasattr(getattr(self.base_dist, key), "to"):
                setattr(self.base_dist, key, getattr(self.base_dist, key).to(device))
