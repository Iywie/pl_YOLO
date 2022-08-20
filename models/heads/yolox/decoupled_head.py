import math
import torch
import torch.nn as nn
from models.layers.network_blocks import BaseConv


class DecoupledHead(nn.Module):
    def __init__(
            self,
            num_classes=80,
            n_anchors=1,
            in_channels=None,
            norm='bn',
            act="silu",
    ):
        super().__init__()
        self.n_anchors = n_anchors
        self.num_classes = num_classes
        ch = self.n_anchors * self.num_classes
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
                BaseConv(in_channels[i], in_channels[0], ksize=1, stride=1, act=act)
            )

            self.cls_convs.append(
                nn.Sequential(
                    *[
                        conv(in_channels[0], in_channels[0], ksize=3, stride=1, norm=norm, act=act),
                        conv(in_channels[0], in_channels[0], ksize=3, stride=1, norm=norm, act=act),
                    ]
                )
            )

            self.cls_preds.append(
                nn.Conv2d(in_channels[0], ch, kernel_size=(1, 1), stride=(1, 1), padding=0)
            )

            self.reg_convs.append(
                nn.Sequential(
                    *[
                        conv(in_channels[0], in_channels[0], ksize=3, stride=1, norm=norm, act=act),
                        conv(in_channels[0], in_channels[0], ksize=3, stride=1, norm=norm, act=act),
                    ]
                )
            )

            self.reg_preds.append(
                nn.Conv2d(in_channels[0], self.n_anchors * 4, kernel_size=(1, 1), stride=(1, 1), padding=0)
            )

            self.obj_preds.append(
                nn.Conv2d(in_channels[0], self.n_anchors * 1, kernel_size=(1, 1), stride=(1, 1), padding=0)
            )

        self.initialize_biases(1e-2)

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
        for k, (cls_conv, reg_conv, x) in enumerate(zip(self.cls_convs, self.reg_convs, inputs)):
            # Change all inputs to the same channel.
            x = self.stems[k](x)

            cls_x = x
            reg_x = x

            cls_feat = cls_conv(cls_x)
            cls_output = self.cls_preds[k](cls_feat)
            reg_feat = reg_conv(reg_x)
            reg_output = self.reg_preds[k](reg_feat)
            obj_output = self.obj_preds[k](reg_feat)

            # output: [batch_size, n_ch, h, w]
            output = torch.cat([reg_output, obj_output, cls_output], 1)
            outputs.append(output)
        return outputs
