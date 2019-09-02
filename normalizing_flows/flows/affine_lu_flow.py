import numpy as np

import torch
import torch.nn as nn
import scipy.linalg as splin

from .flow import Flow

class AffineLUFlow(Flow):
    def __init__(self, dim):
        super(AffineLUFlow, self).__init__()

        W = torch.rand(dim, dim)
        nn.init.orthogonal_(W)
        P, L, U = splin.lu(W)

        shift = nn.init.xavier_uniform_(torch.rand(dim, 1)).squeeze()
        self.shift = nn.Parameter(shift)

        self.register_buffer('P', torch.Tensor(P))
        self.register_buffer('tril_mask', torch.zeros(dim, dim))
        self.register_buffer('triu_mask', torch.zeros(dim, dim))

        tril_indices = np.tril_indices(dim, -1)
        triu_indices = np.triu_indices(dim, 1)

        self.tril_mask[tril_indices] = 1.
        self.triu_mask[triu_indices] = 1.

        s = torch.tensor(U).diag()
        sign_s = np.sign(s.numpy())
        self.register_buffer('sign_s', torch.tensor(sign_s))
        self.log_abs_s = nn.Parameter(s.abs().log())
        #self.s = nn.Parameter(s)

        self.L = nn.Parameter(torch.tensor(L))
        self.U = nn.Parameter(torch.tensor(U))

        self.register_buffer('I', torch.eye(dim))

        if dim == 1:
            self.mul = torch.mul
        else:
            self.mul = torch.matmul

    @property
    def weights(self):
        """
        To accelerate training I'm parameterizing the inverse directly, so
        as to easily calculate log_probs. So self.weights are the weights
        of the inverse transform
        """
        return self.mul(self.P, self.mul(
            self.L * self.tril_mask + self.I,
            #self.U * self.triu_mask + self.s.diag()
            self.U * self.triu_mask + (self.sign_s * self.log_abs_s.exp()).diag()
        ))

    def forward(self, z):
        """
        SEE COMMENT ON THE DEFINITION OF `.weights`
        """
        return self.mul(torch.inverse(self.weights), (z - self.shift).unsqueeze(-1)).squeeze(-1)
        #return self.mul(self.weights, z.unsqueeze(-1)).squeeze(-1) + self.shift

    def inverse(self, x):
        """
        SEE COMMENT ON THE DEFINITION OF `.weights`
        """
        return self.mul(self.weights, x.unsqueeze(-1)).squeeze(-1) + self.shift
        #return self.mul(torch.inverse(self.weights), (x - self.shift).unsqueeze(-1)).squeeze(-1)

    def log_abs_det_jacobian(self, z, z_next):

        #log_abs_det_jac = self.s.abs().log().sum()
        log_abs_det_jac = -self.log_abs_s.sum()

        return (log_abs_det_jac
                    # the jacobian is constant, so we just repeat it for the
                    # whole batch
                    .repeat(z.size(0), 1)
                    .squeeze())

    def to(self, device="cuda:0"):
        super(AffineLUFlow, self).to(device)
