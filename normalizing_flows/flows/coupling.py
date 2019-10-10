import copy
import numpy as np

import torch
import torch.nn as nn

from .flow import Flow

class CouplingLayerFlow(Flow):
    def __init__(self, dim, hidden_size, n_hidden, mask):
        super().__init__()

        self.register_buffer('mask', mask.float())

        # scale function
        s = [nn.Linear(dim, hidden_size)]
        for _ in range(n_hidden):
            s += [nn.ReLU(), nn.Linear(hidden_size, hidden_size)]
        s += [nn.ReLU(), nn.Linear(hidden_size, dim), nn.Tanh()]
        #s += [nn.ReLU(), nn.Linear(hidden_size, dim//2)]
        self.s = nn.Sequential(*s)

        # translation function
        t = [nn.Linear(dim, hidden_size)]
        for _ in range(n_hidden):
            t += [nn.ReLU(), nn.Linear(hidden_size, hidden_size)]
        t += [nn.ReLU(), nn.Linear(hidden_size, dim)]
        self.t = nn.Sequential(*t)

    def inverse(self, x):
        z1 = self.mask * ((x - self.t(x*(1-self.mask))) * torch.exp(-self.s(x*(1-self.mask))))
        z2 = (1 - self.mask) * x

        return z1 + z2

    def forward(self, z):
        x1 = self.mask * (z * torch.exp(self.s(z*(1-self.mask))) + self.t(z*(1-self.mask)))
        x2 = (1-self.mask) * z

        return x1 + x2

    def log_abs_det_jacobian(self, z, z_next=None):
        z2 = (z*(1-self.mask) if z_next is None else z_next*(1-self.mask))

        return self.s(z2).sum(dim=1)

    def to(self, device):
        super(CouplingLayerFlow, self).to(device)

