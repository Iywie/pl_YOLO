import math
import torch
import torch.nn as nn


def YOLOXLoss(outputs, num_classes, strides=None):
    for i in len(outputs):
        # output = torch.cat([reg_output, obj_output, cls_output], 1)
        output, grid = get_output_and_grid(
            outputs[i], i, strides[i], outputs[0].type()
        )



