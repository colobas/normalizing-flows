import copy
import numpy as np

import torch
import torch.nn as nn

from .flow import Flow

class CouplingLayerFlow(Flow):
    def __init__(self, dim, hidden_size, n_hidden, mask):
        super().__init__()

        self.register_buffer('mask', torch.LongTensor(mask))

        self.x1_mask = lambda z: torch.gather(z, 1, torch.nonzero(mask).repeat((len(z),1)))
        self.x2_mask = lambda z: torch.gather(z, 1, torch.nonzero(1-mask).repeat((len(z),1)))

        # scale function
        s = [nn.Linear(dim//2, hidden_size)]
        for _ in range(n_hidden):
            s += [nn.ReLU(), nn.Linear(hidden_size, hidden_size)]
        s += [nn.ReLU(), nn.Linear(hidden_size, dim//2), nn.Tanh()]
        self.s = nn.Sequential(*s)

        # translation function
        t = [nn.Linear(dim//2, hidden_size)]
        for _ in range(n_hidden):
            t += [nn.ReLU(), nn.Linear(hidden_size, hidden_size)]
        t += [nn.ReLU(), nn.Linear(hidden_size, dim//2)]
        self.t = nn.Sequential(*t)

    def forward(self, z):
        # note that z2 = x2
        z1 = self.x1_mask(z)
        x2 = self.x2_mask(z)

        x1 = z1 * (self.s(x2).exp()) + self.t(x2)

        return torch.cat([x1, x2], dim=1)

    def inverse(self, x):
        # note that z2 = x2
        x1 = self.x1_mask(x)
        z2 = self.x2_mask(x)

        z1 = (x1 - self.t(z2)) * ((-self.s(z2)).exp())

        return torch.cat([z1, z2], dim=1)

    def log_abs_det_jacobian(self, z, z_next=None):
        z2 = self.x2_mask(z)
        return self.s(z2).sum(dim=1)

    def to(self, device="cuda:0"):
        super(CouplingLayerFlow, self).to(device)

