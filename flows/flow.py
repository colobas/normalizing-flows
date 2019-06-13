import torch.nn as nn
import torch.distributions as distrib
import torch.distributions.transforms as transform

class Flow(transform.Transform, nn.Module):
    """
    purpose of this class is to make `transform.Transform` 'trainable'
    simple flows will inherit it.

    for an explanation of why I'm setting event_dim=1 as default, see:
    https://github.com/acids-ircam/pytorch_flows/issues/2
    """

    def __init__(self, event_dim=1):
        transform.Transform.__init__(self)
        nn.Module.__init__(self)
        self.bijective = True
        self.event_dim = event_dim

    def init_parameters(self):
        for param in self.parameters():
            param.data.uniform_(-0.01, 0.01)

    def __hash__(self):
        return nn.Module.__hash__(self)
