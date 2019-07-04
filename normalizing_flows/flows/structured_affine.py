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
        self.A_tril = nn.Parameter(torch.randn(int(dim * (dim + 1)/2)))
        self.u = nn.Parameter(torch.randn(dim, 1))
        self.v = nn.Parameter(torch.randn(dim, 1))
        self.shift = nn.Parameter(torch.randn(dim))
        self.A = torch.zeros(dim, dim)
        self.tril_indices = np.tril_indices(dim)

    @property
    def weights(self):
        self.A[self.tril_indices] = self.A_tril

        return self.A + (self.u @ self.v.t())

    def _call(self, z):
        return (self.weights @ z.unsqueeze(-1)).squeeze(-1) + self.shift

    def _inverse(self, z):
        return (torch.inverse(self.weights) @ (z - self.shift).unsqueeze(-1)).squeeze(-1)

    def log_abs_det_jacobian(self, z, z_next):
        """
        Matrix determinant lemma: https://en.wikipedia.org/wiki/Matrix_determinant_lemma
        A is triangular, so its determinant is the product of its diagonal

        det(A + uv^T) = (1 + v^T A^-1 u) * det(A)

        to compute (A^-1 u) use `torch.triangular_solve` (or `torch.trtrs` if 
        `triangular_solve` isn't available)
            - see https://pytorch.org/docs/stable/torch.html#torch.triangular_solve

        """

        self.A[self.tril_indices] = self.A_tril
        det_A = self.A.diag().prod()
        first_term = 1 + self.v.t() @ triangular_solve(self.u, self.A)[0]


        return ((first_term * det_A)
                    # the jacobian is constant, so we just repeat it for the
                    # whole batch
                    .repeat(z.size(0), 1)
                    .squeeze())

