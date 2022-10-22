import torch
from thop import profile, clever_format


def model_summary(model, train_size, device):
    dummy_input = torch.zeros(1, 3, train_size[0], train_size[1]).to(device)
    flops, params = profile(model, inputs=(dummy_input,))
    flops, params = clever_format([flops*2, params], "%.3f")
    print(" ------- params: %s ------- flops: %s" % (params, flops))
    return None
