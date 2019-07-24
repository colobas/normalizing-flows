import numpy as np

import torch
import torch.nn as nn

from .flow import Flow

def make_mask_func(mask):
    mask_func = lambda z: z
    return mask_func


class CouplingLayerFlow(Flow):
    def __init__(self, dim, s, t, x1_mask, x2_mask):
        super(CouplingLayerFlow, self).__init__()

        self.s = s
        self.t = t
        self.dim = dim
        self.x1_mask = make_mask_func(x1_mask)
        self.x2_mask = make_mask_func(x2_mask)

    def _call(self, z):
        # note that z2 = x2
        z1 = self.x1_mask(z)
        x2 = self.x2_mask(z)

        x1 = z1 * (self.s(z2).exp()) + self.t(z2)

        return torch.stack([x1, x2])

    def _inverse(self, x):
        # note that z2 = x2
        x1 = self.x1_mask(x)
        z2 = self.x2_mask(x)

        z1 = (x1 - self.t(z2)) * ((-self.s(z2).exp()))

        return torch.stack([x1, x2])

    def log_abs_det_jacobian(self, z, z_next):
        z2 = self.x2_mask(z)

        return self.s(z2).sum(dim=1).abs()

