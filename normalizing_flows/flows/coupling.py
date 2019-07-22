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
        # note that y1 = x1
        y1 = self.x1_mask(z)
        x2 = self.x2_mask(z)

        y2 = x2 * (self.s(y1).exp()) + self.t(y1)

        return torch.stack([y1, y2])

    def _inverse(self, z):
        # note that y1 = x1
        x1 = self.x1_mask(z)
        y2 = self.x2_mask(z)

        x2 = (y2 - self.t(x1)) * (1/(self.s(y1).exp()))

        return torch.stack([x1, x2])



    def log_abs_det_jacobian(self, z, z_next):

