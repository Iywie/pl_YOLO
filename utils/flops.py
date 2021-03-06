import torch
from thop import profile
from thop import clever_format
from nni.compression.pytorch.utils.counter import count_flops_params


def model_summary(model, train_size, device):
    dummy_input = torch.zeros(1, 3, train_size[0], train_size[1]).to(device)
    # flops, params, results = count_flops_params(model, dummy_input)
    flops, params = profile(model, inputs=(dummy_input,))
    flops, params = clever_format([flops, params], "%.3f")
    print(" ------- params: %s ------- flops: %s" % (params, flops))
    return None
