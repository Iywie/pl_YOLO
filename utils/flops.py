import torch
from thop import profile
from thop import clever_format
from nni.compression.pytorch.utils.counter import count_flops_params


def model_summary(model, train_size):
    dummy_input = torch.zeros(1, 3, train_size[0], train_size[1])
    flops, params, results = count_flops_params(model, dummy_input)
    # macs, params = profile(model, inputs=(dummy_input,))
    # macs, params = clever_format([macs, params], "%.3f")
    return flops, params, results
