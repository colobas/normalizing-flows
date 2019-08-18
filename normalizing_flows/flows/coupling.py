import copy
import numpy as np

import torch
import torch.nn as nn

from .flow import Flow

class CouplingLayerFlow(Flow):
    def __init__(self, input_size, hidden_size, n_hidden, mask):
        super().__init__()

        self.register_buffer('mask', mask)

        # scale function
        s_net = [nn.Linear(input_size, hidden_size)]
        for _ in range(n_hidden):
            s_net += [nn.Tanh(), nn.Linear(hidden_size, hidden_size)]
        s_net += [nn.Tanh(), nn.Linear(hidden_size, input_size)]
        self.s_net = nn.Sequential(*s_net)

        # translation function
        t_net = [nn.Linear(input_size, hidden_size)]
        for _ in range(n_hidden):
            t_net += [nn.ReLU(), nn.Linear(hidden_size, hidden_size)]
        t_net += [nn.ReLU(), nn.Linear(hidden_size, input_size)]
        self.t_net = nn.Sequential(*t_net)

    def forward(self, z):
        # apply mask
        mz = z * self.mask

        # run through model
        s = self.s_net(mz)
        t = self.t_net(mz)
        x = mz + (1 - self.mask) * (z - t) * torch.exp(-s)  # cf RealNVP eq 8 where u corresponds to x (here we're modeling u)

        return x

    def inverse(self, x):
        # apply mask
        mx = x * self.mask

        # run through model
        s = self.s_net(mx)
        t = self.t_net(mx)
        z = mx + (1 - self.mask) * (z * s.exp() + t)  # cf RealNVP eq 7

        return z

    def log_abs_det_jacobian(self, z, x):
        return (1 - self.mask) * self.s_net(mz)
