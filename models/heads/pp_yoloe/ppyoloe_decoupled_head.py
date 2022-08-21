import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.layers.network_blocks import BaseConv


class PPYOLOEDecoupledHead(nn.Module):
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
        # ESE block
        self.stem_cls = nn.ModuleList()
        self.stem_reg = nn.ModuleList()
        self.cls_convs = nn.ModuleList()
        self.cls_preds = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        self.reg_preds = nn.ModuleList()
        self.obj_preds = nn.ModuleList()

        # For each feature map we go through different convolution.
        for i in range(len(in_channels)):
            self.stem_cls.append(ESEAttn(in_channels[i], norm=norm, act=act))
            self.stem_reg.append(ESEAttn(in_channels[i], norm=norm, act=act))

            self.cls_convs.append(
                nn.Sequential(
                    *[
                        conv(in_channels[i], in_channels[i], ksize=3, stride=1, norm=norm, act=act),
                        conv(in_channels[i], in_channels[i], ksize=3, stride=1, norm=norm, act=act),
                    ]
                )
            )
            self.cls_preds.append(
                nn.Conv2d(in_channels[i], ch, kernel_size=(1, 1), stride=(1, 1), padding=0)
            )

            self.reg_convs.append(
                nn.Sequential(
                    *[
                        conv(in_channels[i], in_channels[i], ksize=3, stride=1, norm=norm, act=act),
                        conv(in_channels[i], in_channels[i], ksize=3, stride=1, norm=norm, act=act),
                    ]
                )
            )
            self.reg_preds.append(
                nn.Conv2d(in_channels[i], self.n_anchors * 4, kernel_size=(1, 1), stride=(1, 1), padding=0)
            )

            self.obj_preds.append(
                nn.Conv2d(in_channels[i], self.n_anchors * 1, kernel_size=(1, 1), stride=(1, 1), padding=0)
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
        for k, (cls_conv, reg_conv, x) in enumerate(zip(self.cls_convs, self.reg_convs, inputs)):

            # ECE Block
            avg_x = F.adaptive_avg_pool2d(x, (1, 1))
            cls_x = self.stem_cls[k](x, avg_x) + x
            reg_x = self.stem_reg[k](x, avg_x)

            cls_feat = cls_conv(cls_x)
            cls_output = self.cls_preds[k](cls_feat)
            reg_feat = reg_conv(reg_x)
            reg_output = self.reg_preds[k](reg_feat)
            obj_output = self.obj_preds[k](reg_feat)

            # output: [batch_size, n_ch, h, w]
            output = torch.cat([reg_output, obj_output, cls_output], 1)
            outputs.append(output)
        return outputs


class ImplicitA(nn.Module):
    def __init__(self, channel):
        super(ImplicitA, self).__init__()
        self.channel = channel
        self.implicit = nn.Parameter(torch.zeros(1, channel, 1, 1))
        nn.init.normal_(self.implicit, std=.02)

    def forward(self):
        return self.implicit


class ImplicitM(nn.Module):
    def __init__(self, channel):
        super(ImplicitM, self).__init__()
        self.channel = channel
        self.implicit = nn.Parameter(torch.ones(1, channel, 1, 1))
        nn.init.normal_(self.implicit, mean=1., std=.02)

    def forward(self):
        return self.implicit


class ESEAttn(nn.Module):
    def __init__(self, feat_channels, norm='bn', act='swish'):
        super(ESEAttn, self).__init__()
        self.fc = nn.Conv2d(feat_channels, feat_channels, 1)
        self.conv = BaseConv(feat_channels, feat_channels, 1, stride=1, norm=norm, act=act)

    def forward(self, feat, avg_feat):
        weight = torch.sigmoid(self.fc(avg_feat))
        return self.conv(feat * weight)