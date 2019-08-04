import numpy as np

import torch
import torch.nn as nn

from .flow import Flow

if hasattr(torch, "triangular_solve"):
    triangular_solve = torch.triangular_solve
elif hasattr(torch, "trtrs"):
    triangular_solve = torch.trtrs
else:
    raise NotImplemented("Neither `triangular_solve` nor `trtrs` found in torch")

class StructuredAffineFlow(Flow):
    def __init__(self, dim):
        super(StructuredAffineFlow, self).__init__()

        self.V = nn.Parameter(torch.rand(dim, dim))
        nn.init.xavier_uniform_(self.V)

        shift = nn.init.xavier_uniform_(torch.rand(dim, 1)).squeeze()
        self.shift = nn.Parameter(shift)

        self.tril_mask = torch.zeros(dim, dim)
        self.tril_indices = np.tril_indices(dim)
        self.tril_mask[self.tril_indices] = 1.

        L = torch.zeros(dim, dim)
        L[self.tril_indices] = nn.init.xavier_uniform_(
                torch.ones(len(self.tril_indices[0]), 1)).squeeze()
        self.L = nn.Parameter(L)

        self.I = torch.eye(dim)

        if dim == 1:
            self.mul = torch.mul
        else:
            self.mul = torch.matmul

    @property
    def weights(self):
        return (self.L * self.tril_mask) + self.mul(self.V, self.V.t())

    def _call(self, z):
        return self.mul(self.weights, z.unsqueeze(-1)).squeeze(-1) + self.shift

    def _inverse(self, z):
        return self.mul(torch.inverse(self.weights), (z - self.shift).unsqueeze(-1)).squeeze(-1)

    def log_abs_det_jacobian(self, z, z_next):
        """
        roughly following tensorflow's LinearOperatorLowRankUpdate logic and
        naming, but U and V are the same, and D is the identity
        """
        log_det_L = self.L.diag().abs().log().sum()

        linv_u = triangular_solve(self.V, self.L * self.tril_mask, upper=False)[0]
        vt_linv_u = self.V.t() @ linv_u
        capacitance = vt_linv_u + self.I

        return ((log_det_L + capacitance.det().abs().log()) # we would sum the log determinant of the diagonal component, but it is 0, because the diagnoal is I
                    # the jacobian is constant, so we just repeat it for the
                    # whole batch
                    .repeat(z.size(0), 1)
                    .squeeze())

