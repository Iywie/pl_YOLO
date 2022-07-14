import torch.nn as nn


def get_normalization(name, out_channels):
    if name is None:
        return None
    if name == "bn":
        module = nn.BatchNorm2d(out_channels, eps=1e-3, momentum=0.03)
    elif name == "ln":
        module = nn.LayerNorm(out_channels)
    else:
        raise AttributeError("Unsupported normalization function type: {}".format(name))
    return module
