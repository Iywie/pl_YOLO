import math
import torch
import torch.nn as nn
from models.layers.network_blocks import BaseConv
from torchvision.ops import deform_conv2d
from models.layers.attention import SKFF


class YOLOXSADecoupledHead(nn.Module):
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
        self.attention_cls = nn.ModuleList()
        self.attention_reg = nn.ModuleList()
        self.cls_convs = nn.ModuleList()
        self.cls_preds = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        self.reg_preds = nn.ModuleList()
        self.obj_preds = nn.ModuleList()
        self.implicitA = nn.ModuleList()
        self.implicitM = nn.ModuleList()
        self.cls_prob_conv1 = nn.ModuleList()
        self.cls_prob_conv2 = nn.ModuleList()
        self.cls_attention = nn.ModuleList()
        self.reg_offset_conv1 = nn.ModuleList()
        self.reg_offset_conv2 = nn.ModuleList()

        # For each feature map we go through different convolution.
        for i in range(len(in_channels)):
            self.implicitA.append(
                ImplicitA(in_channels[i])
            )
            self.stems.append(
                BaseConv(in_channels[i], in_channels[0], ksize=1, stride=1, act=act)
            )
            self.implicitM.append(
                ImplicitM(in_channels[0])
            )
            # self.attention_cls.append(SALayer(in_channels[0], groups=int(64)))
            # self.attention_reg.append(SALayer(in_channels[0], groups=int(64)))

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

            self.cls_prob_conv1.append(
                conv(in_channels[0], in_channels[0], ksize=1, stride=1, norm=norm, act=act)
            )
            self.cls_prob_conv2.append(
                nn.Conv2d(in_channels[0], ch, kernel_size=(3, 3), stride=(1, 1), padding=1)
            )
            self.cls_attention.append(
                SKFF(ch, height=2)
            )
            self.reg_offset_conv1.append(
                conv(in_channels[0], in_channels[0], ksize=1, stride=1, norm=norm, act=act)
            )
            self.reg_offset_conv2.append(
                nn.Conv2d(in_channels[0], self.n_anchors * 8, kernel_size=(3, 3), stride=(1, 1), padding=1)
            )
            self.relu = nn.ReLU(inplace=True)

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
            x = x + self.implicitA[k]().expand_as(x)
            # Change all inputs to the same channel.
            x = self.stems[k](x)
            x = x * self.implicitM[k]().expand_as(x)

            # cls_x = self.attention_cls[k](x)
            # reg_x = self.attention_reg[k](x)
            cls_x = x
            reg_x = x

            # M = self.cls_prob_conv1[k](x)
            # M = self.cls_prob_conv2[k](M)

            cls_feat = cls_conv(cls_x)
            cls_output = self.cls_preds[k](cls_feat)
            # cls_output = self.cls_attention[k]([M, cls_output])
            # cls_output = (cls_output.sigmoid() * M.sigmoid()).sqrt()
            # cls_output = torch.log(cls_output / (1 - cls_output + 1e-16) + 1e-16)

            # b, c, h, w = cls_output.shape
            # weight = cls_output.new_ones(c, 1, 1, 1)
            # cls_output = deform_conv2d(cls_output, M, weight, mask=None)

            O = self.reg_offset_conv1[k](cls_feat)
            O = self.reg_offset_conv2[k](O)

            reg_feat = reg_conv(reg_x)
            obj_output = self.obj_preds[k](reg_feat)
            b, c, h, w = reg_feat.shape
            weight = reg_feat.new_ones(c, 1, 1, 1)
            reg_feat = deform_conv2d(reg_feat, O, weight, mask=None)
            # reg_feat = self.relu(reg_feat)
            reg_output = self.reg_preds[k](reg_feat)
            # b, c, h, w = reg_output.shape
            # weight = reg_output.new_ones(c, 1, 1, 1)
            # reg_output = deform_conv2d(reg_output, O, weight, mask=None)

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


class SALayer(nn.Module):
    """
    Constructs a Channel Spatial Group module.
    """

    def __init__(self, channel, groups=64):
        super(SALayer, self).__init__()
        self.groups = groups
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.cweight = nn.Parameter(torch.zeros(1, channel // (2 * groups), 1, 1))
        self.cbias = nn.Parameter(torch.ones(1, channel // (2 * groups), 1, 1))
        self.sweight = nn.Parameter(torch.zeros(1, channel // (2 * groups), 1, 1))
        self.sbias = nn.Parameter(torch.ones(1, channel // (2 * groups), 1, 1))

        self.sigmoid = nn.Sigmoid()
        self.gn = nn.GroupNorm(channel // (2 * groups), channel // (2 * groups))

    @staticmethod
    def channel_shuffle(x, groups):
        b, c, h, w = x.shape

        x = x.reshape(b, groups, -1, h, w)
        x = x.permute(0, 2, 1, 3, 4)

        # flatten
        x = x.reshape(b, -1, h, w)

        return x

    def forward(self, x):
        b, c, h, w = x.shape

        x = x.reshape(b * self.groups, -1, h, w)
        x_0, x_1 = x.chunk(2, dim=1)

        # channel attention
        xn = self.avg_pool(x_0)
        xn = self.cweight * xn + self.cbias
        xn = x_0 * self.sigmoid(xn)

        # spatial attention
        xs = self.gn(x_1)
        xs = self.sweight * xs + self.sbias
        xs = x_1 * self.sigmoid(xs)

        # concatenate along channel axis
        out = torch.cat([xn, xs], dim=1)
        out = out.reshape(b, -1, h, w)

        out = self.channel_shuffle(out, 2)
        return out
