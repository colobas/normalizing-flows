import torch
import torch.nn as nn

from .flow import Flow

class AffineFlow(Flow):
    def __init__(self, dim):
        super(AffineFlow, self).__init__()
        self.weights = nn.Parameter(torch.eye(dim))
        self.shift = nn.Parameter(torch.zeros(dim))
        nn.init.orthogonal_(self.weights)

    def _call(self, z):
        return (self.weights @ z.unsqueeze(-1)).squeeze(-1) + self.shift

    def _inverse(self, z):
        return (torch.inverse(self.weights) @ (z - self.shift).unsqueeze(-1)).squeeze(-1)

    def log_abs_det_jacobian(self, z, z_next):
        return self.weights.det().abs().log().unsqueeze(0).repeat(z.size(0), 1).squeeze()

