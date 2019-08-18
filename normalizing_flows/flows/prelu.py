import torch
import torch.nn as nn
import torch.distributions as distrib
import torch.distributions.transforms as transform

from .flow import Flow

class PReLUFlow(Flow):
    def __init__(self, dim):
        super(PReLUFlow, self).__init__()
        self.alpha = nn.Parameter(torch.Tensor([0.5]))
        self.bijective = True

        if dim == 1:
            self.agg_return = lambda res: res
        else:
            self.agg_return = lambda res: torch.sum(res, dim=1)

    def forward(self, z):
        return torch.where(z >= 0, z, self.alpha * z)

    def inverse(self, x):
        return torch.where(x >= 0, x, (1. / self.alpha) * x)

    def log_abs_det_jacobian(self, z, x):
        I = torch.ones_like(z)
        J = torch.where(z >= 0, I, self.alpha * I)
        log_abs_det = torch.log(torch.abs(J) + 1e-5)
        return self.agg_return(log_abs_det)
