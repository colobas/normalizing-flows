import numpy as np

import torch
import torch.nn as nn

from .flow import Flow

def make_mask_func(mask):
    mask = torch.LongTensor(mask)

    def mask_func(x, dim=1):
        # based on https://discuss.pytorch.org/t/batched-index-select/9115/10
        views = [1 if i != dim else -1 for i in range(len(x.shape))]
        expanse = list(x.shape)
        expanse[dim] = -1
        return torch.gather(x, dim, mask.view(views).expand(expanse))

    return mask_func

class CouplingLayerFlow(Flow):
    def __init__(self, dim, s, t, x1_mask):
        super(CouplingLayerFlow, self).__init__()

        self.s = s
        self.t = t
        self.dim = dim

        x2_mask = [i for i in range(dim) if i not in x1_mask]

        self.x1_mask = make_mask_func(x1_mask)
        self.x2_mask = make_mask_func(x2_mask)

    def _call(self, z):
        # note that z2 = x2
        z1 = self.x1_mask(z)
        x2 = self.x2_mask(z)

        x1 = z1 * (self.s(x2).exp()) + self.t(x2)

        return torch.cat([x1, x2], dim=1)

    def _inverse(self, x):
        # note that z2 = x2
        x1 = self.x1_mask(x)
        z2 = self.x2_mask(x)

        z1 = (x1 - self.t(z2)) * ((-self.s(z2).exp()))

        return torch.cat([z1, z2], dim=1)

    def log_abs_det_jacobian(self, z, z_next):
        z2 = self.x2_mask(z)

        return self.s(z2).sum(dim=1).abs()
