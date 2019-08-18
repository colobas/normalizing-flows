import torch.nn as nn
import torch.distributions as distrib
import torch.distributions.transforms as transform

class Flow(nn.Module):
    def __hash__(self):
        return nn.Module.__hash__(self)

    def forward(self, z):
        raise NotImplemented

    def inverse(self, x):
        raise NotImplemented

    def log_abs_det_jacobian(self, z, x):
        raise NotImplemented
