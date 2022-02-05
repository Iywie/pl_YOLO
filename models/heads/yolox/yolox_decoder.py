import math
import torch
import torch.nn as nn
from models.layers.network_blocks import BaseConv


class YOLOXDecoder(nn.Module):
    def __init__(
            self,
            num_classes,
            strides=None,
            in_channels=None,
            norm='bn',
            act="silu",
    ):
        super().__init__()
        self.n_anchors = 1
        self.num_classes = num_classes
        self.strides = strides
        self.grids = [torch.zeros(1)] * len(in_channels)
        conv = BaseConv

        self.stems = nn.ModuleList()
        self.cls_convs = nn.ModuleList()
        self.cls_preds = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        self.reg_preds = nn.ModuleList()
        self.obj_preds = nn.ModuleList()

        # For each feature map we go through different convolution.
        for i in range(len(in_channels)):
            self.stems.append(
                BaseConv(
                    in_channels=int(in_channels[i]),
                    out_channels=int(256),
                    ksize=1,
                    stride=1,
                    act=act,
                )
            )

            self.cls_convs.append(
                nn.Sequential(
                    *[
                        conv(256, 256, ksize=3, stride=1, norm=norm, act=act),
                        conv(256, 256, ksize=3, stride=1, norm=norm, act=act),
                    ]
                )
            )

            self.cls_preds.append(
                nn.Conv2d(
                    in_channels=int(256),
                    out_channels=self.n_anchors * self.num_classes,
                    kernel_size=(1, 1),
                    stride=(1, 1),
                    padding=0,
                )
            )

            self.reg_convs.append(
                nn.Sequential(
                    *[
                        conv(int(256), 256, ksize=3, stride=1, norm=norm, act=act),
                        conv(int(256), 256, ksize=3, stride=1, norm=norm, act=act),
                    ]
                )
            )

            self.reg_preds.append(
                nn.Conv2d(
                    in_channels=int(256),
                    out_channels=4,
                    kernel_size=(1, 1),
                    stride=(1, 1),
                    padding=0,
                )
            )

            self.obj_preds.append(
                nn.Conv2d(
                    in_channels=int(256),
                    out_channels=self.n_anchors * 1,
                    kernel_size=(1, 1),
                    stride=(1, 1),
                    padding=0,
                )
            )

    def initialize_biases(self, prior_prob):
        for conv in self.cls_preds:
            b = conv.bias.view(self.n_anchors, -1)
            b.data.fill_(-math.log((1 - prior_prob) / prior_prob))
            conv.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

        for conv in self.obj_preds:
            b = conv.bias.view(self.n_anchors, -1)
            b.data.fill_(-math.log((1 - prior_prob) / prior_prob))
            conv.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

    def forward(self, inputs):
        outputs = []
        x_shifts = []
        y_shifts = []
        expanded_strides = []
        batch_size = inputs[0].shape[0]
        n_ch = 4 + 1 + self.n_anchors * self.num_classes  # the channel of one ground truth prediction.
        data_type = inputs[0].type()

        for k, (cls_conv, reg_conv, stride_this_level, x) in enumerate(
                zip(self.cls_convs, self.reg_convs, self.strides, inputs)
        ):
            # Change all inputs to the same channel. (like 256)
            x = self.stems[k](x)

            cls_feat = cls_conv(x)
            cls_output = self.cls_preds[k](cls_feat)
            reg_feat = reg_conv(x)
            reg_output = self.reg_preds[k](reg_feat)
            obj_output = self.obj_preds[k](reg_feat)

            # output: [batch_size, n_ch, h, w]
            output = torch.cat([reg_output, obj_output, cls_output], 1)
            h, w = output.shape[-2:]

            # Three steps to localize predictions: grid, shifts of x and y, grid with stride
            grid = self.get_grid(output, k)
            x_shifts.append(grid[:, :, 0])
            y_shifts.append(grid[:, :, 1])
            expanded_strides.append(
                torch.zeros(1, grid.shape[1]).fill_(stride_this_level)
            )

            output = output.view(batch_size, self.n_anchors, n_ch, h, w)
            output = output.permute(0, 1, 3, 4, 2).reshape(
                batch_size, self.n_anchors * h * w, -1
            )
            # output: [batch_size, h * w, n_ch]
            grid = grid.view(1, -1, 2)
            # grid: [1, h * w, 2]
            output[..., :2] = (output[..., :2] + grid) * stride_this_level
            # The predictions of w and y are logs
            output[..., 2:4] = torch.exp(output[..., 2:4]) * stride_this_level
            outputs.append(output)

        # outputs: [batch_size, all layer predictions, n_ch]
        outputs = torch.cat(outputs, 1)
        return outputs, x_shifts, y_shifts, expanded_strides

    def get_grid(self, output, k):
        h, w = output.shape[-2:]
        yv, xv = torch.meshgrid([torch.arange(h), torch.arange(w)])
        grid = torch.stack((xv, yv), 2).view(1, 1, h, w, 2)
        grid = grid.view(1, -1, 2)
        self.grids[k] = grid
        return grid
