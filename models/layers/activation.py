import torch.nn as nn
import torch.nn.functional as F


def get_activation(name="silu", inplace=True):
    if name is None:
        return None
    if name == "silu":
        module = nn.SiLU(inplace=inplace)
    elif name == "relu":
        module = nn.ReLU(inplace=inplace)
    elif name == "lrelu":
        module = nn.LeakyReLU(0.1, inplace=inplace)
    elif name == 'hswish':
        module = HSwish()
    elif name == "gelu":
        module = nn.GELU()
    else:
        raise AttributeError("Unsupported activation function type: {}".format(name))
    return module


class HSwish(nn.Module):

    def forward(self, x):
        out = x * F.relu6(x + 3, inplace=True) / 6
        return out
