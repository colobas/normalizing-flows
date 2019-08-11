import torch.nn as nn
import torch.distributions as distrib
import torch.distributions.transforms as transform

class NormalizingFlow(nn.Module):
    """
    Instantiate a trainable NormalizingFlow

    dim: random variable dimension
    blocks: list of primitive flows to stack
    flow_length: how many times to stack the blocks in `blocks`

    example:
        - to have 3 x [AffineLUFlow, PReLUFlow], do
            blocks=[AffineLUFlow, PReLUFlow],
            flow_length=3

        - to specify the sequence of flows explicitely, do
            blocks=[the, specific, sequence, of, flows, you, want],
            flow_length=1

    `flow_length` defaults to 1, because I think the latter scenario
    is what most people will assume without reading the docstring
    """

    def __init__(self, dim, blocks, base_density, flow_length=1, block_args=None):
        super().__init__()
        biject = []
        if block_args is None:
            for f in range(flow_length):
                for b_flow in blocks:
                    biject.append(b_flow(dim))
        else:
             for f in range(flow_length):
                for b_flow, b_args in zip(blocks, block_args):
                    if b_args is not None:
                        biject.append(b_flow(dim, *b_args))
                    else:
                        biject.append(b_flow(dim))

        self.transforms = transform.ComposeTransform(biject)
        self.bijectors = nn.ModuleList(biject)
        self.base_density = base_density
        self.final_density = distrib.TransformedDistribution(base_density, self.transforms)

    def forward(self, z0):
        log_det = []
        # Applies series of flows
        z_cur = z0
        for b in range(len(self.bijectors)):
            z_next = self.bijectors[b](z_cur)
            log_det.append(self.bijectors[b].log_abs_det_jacobian(z_cur, z_next))
            z_cur = z_next
        return z_next, log_det

    def inverse(self, z):
        L = len(self.bijectors)
        z_cur = z

        for b in range(L):
            z_cur = self.bijectors[(L-1) - b]._inverse(z_cur)
            self.log_det.append(self.bijectors[b].log_abs_det_jacobian(z_cur, z_next))

        return z_cur

    def log_prob(self, x):
        return self.final_density.log_prob(x)

    def to(self, device="cuda:0"):
        super(NormalizingFlow, self).to(device)
        for bij in self.bijectors:
            bij.to(device)

        for key in self.base_density.__dict__:
            if hasattr(getattr(self.base_density, key), "to"):
                setattr(self.base_density, key, getattr(self.base_density, key).to(device))
