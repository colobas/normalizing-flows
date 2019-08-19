import torch
import torch.nn as nn
import torch.distributions as distrib
import torch.distributions.transforms as transform

class NormalizingFlow(nn.Sequential):
    def __init__(self, *args, base_dist):
        super(NormalizingFlow, self).__init__(*args)
        self.base_dist = base_dist

    def forward(self, z_0):
        sum_log_abs_det = 0
        # Applies series of flows
        z_cur = z_0
        for flow in self:
            z_next = flow(z_cur)
            sum_log_abs_det = sum_log_abs_det - flow.log_abs_det_jacobian(z_cur, z_next)
            z_cur = z_next
        return z_next, sum_log_abs_det

    def inverse(self, x):
        sum_log_abs_det = 0
        z_cur = x
        for flow in reversed(self):
            z_prev = flow.inverse(z_cur)
            sum_log_abs_det = sum_log_abs_det - flow.log_abs_det_jacobian(z_prev, z_cur)
            z_cur = z_prev

        return z_prev, sum_log_abs_det

    def log_prob(self, x):
        z, log_abs_det_jacobian = self.inverse(x)

        base_log_probs = self.base_dist.log_prob(z)

        if len(base_log_probs.shape) > 1:
            return base_log_probs.sum(dim=1) + log_abs_det_jacobian
        else:
            return base_log_probs + log_abs_det_jacobian

    def sample(self, n_samples):
        with torch.no_grad():
            z_samples = self.base_dist.sample((n_samples, ))
            return self.forward(z_samples)[0]

    def to(self, device="cuda:0"):
        super(NormalizingFlow, self).to(device)

        for flow in self:
            flow.to(device)

        for key in self.base_dist.__dict__:
            if hasattr(getattr(self.base_dist, key), "to"):
                setattr(self.base_dist, key, getattr(self.base_dist, key).to(device))
