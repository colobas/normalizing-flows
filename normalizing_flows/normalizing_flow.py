import torch.nn as nn
import torch.distributions as distrib
import torch.distributions.transforms as transform

class NormalizingFlow(nn.Module):

    def __init__(self, dim, blocks, flow_length, density):
        super().__init__()
        biject = []
        for f in range(flow_length):
            for b_flow in blocks:
                biject.append(b_flow(dim))
        self.transforms = transform.ComposeTransform(biject)
        self.bijectors = nn.ModuleList(biject)
        self.base_density = density
        self.final_density = distrib.TransformedDistribution(density, self.transforms)
        self.log_det = []

    def forward(self, z0):
        self.log_det = []
        # Applies series of flows
        z_cur = z0
        for b in range(len(self.bijectors)):
            z_next = self.bijectors[b](z_cur)
            self.log_det.append(self.bijectors[b].log_abs_det_jacobian(z_cur, z_next))
            z_cur = z_next
        return z_next, self.log_det
