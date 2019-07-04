import torch
import torch.nn as nn
import torch.distributions as distrib
import torch.distributions.transforms as transform

from .flow import Flow

class PReLUFlow(Flow):
    def __init__(self, dim):
        super(PReLUFlow, self).__init__()
        self.alpha = nn.Parameter(torch.Tensor([1]))
        self.bijective = True
        #self.init_parameters()

    def init_parameters(self):
        for param in self.parameters():
            param.data.uniform_(0.01, 0.5)

    def _call(self, z):
        return torch.where(z >= 0, z, self.alpha * z)

    def _inverse(self, z):
        return torch.where(z >= 0, z, (1. / self.alpha) * z)

    def log_abs_det_jacobian(self, z_cur, z_next):
        I = torch.ones_like(z_cur)
        J = torch.where(z_cur >= 0, I, self.alpha * I)
        log_abs_det = torch.log(torch.abs(J) + 1e-5)
        return torch.sum(log_abs_det, dim=1)
