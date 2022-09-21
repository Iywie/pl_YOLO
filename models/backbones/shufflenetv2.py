"""
ShuffleNetV2+
Depths: [4, 4, 8, 4]
Architecture: [0, 0, 3, 1, 1, 1, 0, 0, 2, 0, 2, 1, 1, 0, 2, 0, 2, 1, 3, 2]
Channels:
    Small:  [36, 104, 208, 416]
    Medium: [48, 128, 256, 512]
    Large:  [68, 168, 336, 672]

"""

import torch
import torch.nn as nn
from models.layers.network_blocks import BaseConv
from models.layers.activation import HSwish
from models.layers.attention import SELayer


class ShuffleNetV2_Plus(nn.Module):
    def __init__(
            self,
            channels=(36, 104, 208, 416),
            out_features=("stage2", "stage3", "stage4"),
            norm='bn',
            act="silu",
    ):
        super(ShuffleNetV2_Plus, self).__init__()
        self.out_features = out_features

        # building first layer
        self.stem = BaseConv(3, 16, 3, 2, norm="bn", act="hswish")

        useSE = False
        self.stage1 = nn.Sequential(
            Shufflenet(16, channels[0], ksize=3, stride=2, activation=act, useSE=useSE),
            Shufflenet(channels[0] // 2, channels[0], ksize=3, stride=1, activation=act, useSE=useSE),
            Shuffle_Xception(channels[0] // 2, channels[0], stride=1, activation=act, useSE=useSE),
            Shufflenet(channels[0] // 2, channels[0], ksize=5, stride=1, activation=act, useSE=useSE),
        )

        act = "hswish"
        useSE = False
        self.stage2 = nn.Sequential(
            Shufflenet(channels[0], channels[1], ksize=3, stride=2, activation=act, useSE=useSE),
            Shufflenet(channels[1] // 2, channels[1], ksize=3, stride=1, activation=act, useSE=useSE),
            Shuffle_Xception(channels[1] // 2, channels[1], stride=1, activation=act, useSE=useSE),
            Shuffle_Xception(channels[1] // 2, channels[1], stride=1, activation=act, useSE=useSE),
        )

        act = "hswish"
        useSE = True
        self.stage3 = nn.Sequential(
            Shufflenet(channels[1], channels[2], ksize=7, stride=2, activation=act, useSE=useSE),
            Shufflenet(channels[2] // 2, channels[2], ksize=3, stride=1, activation=act, useSE=useSE),
            Shufflenet(channels[2] // 2, channels[2], ksize=7, stride=1, activation=act, useSE=useSE),
            Shufflenet(channels[2] // 2, channels[2], ksize=5, stride=1, activation=act, useSE=useSE),
            Shufflenet(channels[2] // 2, channels[2], ksize=5, stride=1, activation=act, useSE=useSE),
            Shufflenet(channels[2] // 2, channels[2], ksize=3, stride=1, activation=act, useSE=useSE),
            Shufflenet(channels[2] // 2, channels[2], ksize=7, stride=1, activation=act, useSE=useSE),
            Shufflenet(channels[2] // 2, channels[2], ksize=3, stride=1, activation=act, useSE=useSE),
        )

        act = "hswish"
        useSE = True
        self.stage4 = nn.Sequential(
            Shufflenet(channels[2], channels[3], ksize=7, stride=2, activation=act, useSE=useSE),
            Shufflenet(channels[3] // 2, channels[3], ksize=5, stride=1, activation=act, useSE=useSE),
            Shuffle_Xception(channels[3] // 2, channels[3], stride=1, activation=act, useSE=useSE),
            Shufflenet(channels[3] // 2, channels[3], ksize=7, stride=1, activation=act, useSE=useSE),
        )

        self._initialize_weights()

    def forward(self, x):
        outputs = {}
        x = self.stem(x)
        outputs["stem"] = x
        x = self.stage1(x)
        outputs["stage1"] = x
        x = self.stage2(x)
        outputs["stage2"] = x
        x = self.stage3(x)
        outputs["stage3"] = x
        x = self.stage4(x)
        outputs["stage4"] = x
        if len(self.out_features) <= 1:
            return x
        return [v for k, v in outputs.items() if k in self.out_features]

    def _initialize_weights(self):
        for name, m in self.named_modules():
            if isinstance(m, nn.Conv2d):
                if 'first' in name or 'SE' in name:
                    nn.init.normal_(m.weight, 0, 0.01)
                else:
                    nn.init.normal_(m.weight, 0, 1.0 / m.weight.shape[1])
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0001)
                nn.init.constant_(m.running_mean, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0001)
                nn.init.constant_(m.running_mean, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


class Shufflenet(nn.Module):

    def __init__(self, in_ch, out_ch, *, ksize, stride, activation, useSE):
        super(Shufflenet, self).__init__()
        self.stride = stride

        base_mid_channels = out_ch // 2
        pad = ksize // 2
        outputs = out_ch - in_ch

        branch_main = [
            # pw
            nn.Conv2d(in_ch, base_mid_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(base_mid_channels),
            None,
            # dw
            nn.Conv2d(base_mid_channels, base_mid_channels, ksize, stride, pad, groups=base_mid_channels, bias=False),
            nn.BatchNorm2d(base_mid_channels),
            # pw-linear
            nn.Conv2d(base_mid_channels, outputs, 1, 1, 0, bias=False),
            nn.BatchNorm2d(outputs),
            None,
        ]
        if activation == 'relu':
            assert useSE is False
            '''This model should not have SE with ReLU'''
            branch_main[2] = nn.ReLU(inplace=True)
            branch_main[-1] = nn.ReLU(inplace=True)
        else:
            branch_main[2] = HSwish()
            branch_main[-1] = HSwish()
            if useSE:
                branch_main.append(SELayer(outputs))
        self.branch_main = nn.Sequential(*branch_main)

        if stride == 2:
            branch_proj = [
                # dw
                nn.Conv2d(in_ch, in_ch, ksize, stride, pad, groups=in_ch, bias=False),
                nn.BatchNorm2d(in_ch),
                # pw-linear
                nn.Conv2d(in_ch, in_ch, 1, 1, 0, bias=False),
                nn.BatchNorm2d(in_ch),
                None,
            ]
            if activation == 'relu':
                branch_proj[-1] = nn.ReLU(inplace=True)
            else:
                branch_proj[-1] = HSwish()
            self.branch_proj = nn.Sequential(*branch_proj)
        else:
            self.branch_proj = None

    def forward(self, old_x):
        if self.stride == 1:
            x_proj, x = channel_shuffle(old_x)
            return torch.cat((x_proj, self.branch_main(x)), 1)
        elif self.stride == 2:
            x_proj = old_x
            x = old_x
            return torch.cat((self.branch_proj(x_proj), self.branch_main(x)), 1)


class Shuffle_Xception(nn.Module):

    def __init__(self, in_ch, out_ch, *, stride, activation, useSE):
        super(Shuffle_Xception, self).__init__()

        self.stride = stride
        base_mid_channel = out_ch // 2
        ksize = 3
        pad = 1
        inp = in_ch
        outputs = out_ch - in_ch

        branch_main = [
            # dw
            nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
            nn.BatchNorm2d(inp),
            # pw
            nn.Conv2d(inp, base_mid_channel, 1, 1, 0, bias=False),
            nn.BatchNorm2d(base_mid_channel),
            None,
            # dw
            nn.Conv2d(base_mid_channel, base_mid_channel, 3, stride, 1, groups=base_mid_channel, bias=False),
            nn.BatchNorm2d(base_mid_channel),
            # pw
            nn.Conv2d(base_mid_channel, base_mid_channel, 1, 1, 0, bias=False),
            nn.BatchNorm2d(base_mid_channel),
            None,
            # dw
            nn.Conv2d(base_mid_channel, base_mid_channel, 3, stride, 1, groups=base_mid_channel, bias=False),
            nn.BatchNorm2d(base_mid_channel),
            # pw
            nn.Conv2d(base_mid_channel, outputs, 1, 1, 0, bias=False),
            nn.BatchNorm2d(outputs),
            None,
        ]

        if activation == 'relu':
            branch_main[4] = nn.ReLU(inplace=True)
            branch_main[9] = nn.ReLU(inplace=True)
            branch_main[14] = nn.ReLU(inplace=True)
        else:
            branch_main[4] = HSwish()
            branch_main[9] = HSwish()
            branch_main[14] = HSwish()
        assert None not in branch_main

        if useSE:
            assert activation != 'relu'
            branch_main.append(SELayer(outputs))

        self.branch_main = nn.Sequential(*branch_main)

        if self.stride == 2:
            branch_proj = [
                # dw
                nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
                nn.BatchNorm2d(inp),
                # pw-linear
                nn.Conv2d(inp, inp, 1, 1, 0, bias=False),
                nn.BatchNorm2d(inp),
                None,
            ]
            if activation == 'relu':
                branch_proj[-1] = nn.ReLU(inplace=True)
            else:
                branch_proj[-1] = HSwish()
            self.branch_proj = nn.Sequential(*branch_proj)

    def forward(self, old_x):
        if self.stride == 1:
            x_proj, x = channel_shuffle(old_x)
            return torch.cat((x_proj, self.branch_main(x)), 1)
        elif self.stride == 2:
            x_proj = old_x
            x = old_x
            return torch.cat((self.branch_proj(x_proj), self.branch_main(x)), 1)


def channel_shuffle(x):
    batchsize, num_channels, height, width = x.data.size()
    assert (num_channels % 4 == 0)
    x = x.reshape(batchsize * num_channels // 2, 2, height * width)
    x = x.permute(1, 0, 2)
    x = x.reshape(2, -1, num_channels // 2, height, width)
    return x[0], x[1]
