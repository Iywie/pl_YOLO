"""
CSPDarkNet
Depths and Channels
    DarkNet-tiny   (1, 3, 3, 1)     (24, 48, 96, 192, 384)
    DarkNet-small  (2, 6, 6, 2)     (32, 64, 128, 256, 512)
    DarkNet-base   (3, 9, 9, 3)     (64, 128, 256, 512, 1024)
    DarkNet-large  (4, 12, 12, 4)   (64, 128, 256, 512, 1024)
"""

import torch
from torch import nn
from models.layers.network_blocks import BaseConv, SPPBottleneck


class EELAN(nn.Module):
    """
    Extended efficient layer aggregation networks (EELAN)
    """
    def __init__(
        self,
        depths=(4, 4, 4, 4),
        channels=(64, 128, 256, 512, 1024),
        out_features=("stage2", "stage3", "stage4"),
        norm='bn',
        act="silu",
    ):
        super().__init__()

        # parameters of the network
        assert out_features, "please provide output features of EELAN!"
        self.out_features = out_features

        # stem
        self.stem = nn.Sequential(
            BaseConv(3, 32, 3, 1, norm=norm, act=act),
            BaseConv(32, channels[0], 3, 2, norm=norm, act=act),
            BaseConv(channels[0], channels[0], 3, 1, norm=norm, act=act),
        )

        # stage1
        self.stage1 = nn.Sequential(
            BaseConv(channels[0], channels[1], 3, 2, norm=norm, act=act),
            CSPLayer(channels[1], channels[2], expansion=0.5, num_bottle=depths[0], norm=norm, act=act),
        )

        # stage2
        self.stage2 = nn.Sequential(
            Transition(channels[2], mpk=2, norm=norm, act=act),
            CSPLayer(channels[2], channels[3], expansion=0.5, num_bottle=depths[1], norm=norm, act=act),
        )

        # stage3
        self.stage3 = nn.Sequential(
            Transition(channels[3], mpk=2, norm=norm, act=act),
            CSPLayer(channels[3], channels[4], expansion=0.5, num_bottle=depths[2], norm=norm, act=act),
        )

        # stage4
        self.stage4 = nn.Sequential(
            Transition(channels[4], mpk=2, norm=norm, act=act),
            SPPBottleneck(channels[4], channels[4], norm=norm, act=act),
            CSPLayer(channels[4], channels[4], expansion=0.5, num_bottle=depths[3], norm=norm, act=act),
        )

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


class CSPLayer(nn.Module):
    def __init__(
        self,
        in_channel,
        out_channel,
        expansion=0.5,
        num_bottle=1,
        norm='bn',
        act="silu",
    ):
        """
        Args:
            in_channel (int): input channels.
            out_channel (int): output channels.
            expansion (float): the number that hidden channels compared with output channels.
            num_bottle (int): number of Bottlenecks. Default value: 1.
            norm (str): type of normalization
            act (str): type of activation
        """
        super().__init__()
        hi_channel = int(in_channel * expansion)  # hidden channels
        self.num_conv = num_bottle//2 if num_bottle > 2 else 1

        self.conv1 = BaseConv(in_channel, hi_channel, 1, stride=1, norm=norm, act=act)
        self.conv2 = BaseConv(in_channel, hi_channel, 1, stride=1, norm=norm, act=act)
        self.conv3 = nn.Sequential(
            *[BaseConv(hi_channel, hi_channel, 3, stride=1, norm=norm, act=act) for _ in range(self.num_conv)]
        )
        self.conv4 = nn.Sequential(
            *[BaseConv(hi_channel, hi_channel, 3, stride=1, norm=norm, act=act) for _ in range(self.num_conv)]
        )

        self.conv5 = BaseConv(4 * hi_channel, out_channel, 1, stride=1, norm=norm, act=act)

    def forward(self, x):
        x_1 = self.conv1(x)
        x_2 = self.conv2(x)
        x_3 = self.conv3(x_2)
        x_4 = self.conv4(x_3)
        x_all = [x_1, x_2, x_3, x_4]
        x = torch.cat(x_all, dim=1)
        return self.conv5(x)


class Transition(nn.Module):
    def __init__(self, in_channel, mpk=2, norm='bn', act="silu"):
        super(Transition, self).__init__()
        self.mp = nn.MaxPool2d(kernel_size=mpk, stride=mpk)
        self.conv1 = BaseConv(in_channel, in_channel//2, 1, 1)
        self.conv2 = BaseConv(in_channel, in_channel//2, 1, 1)
        self.conv3 = BaseConv(in_channel//2, in_channel//2, 3, 2, norm=norm, act=act)

    def forward(self, x):
        x_1 = self.mp(x)
        x_1 = self.conv1(x_1)

        x_2 = self.conv2(x)
        x_2 = self.conv3(x_2)

        return torch.cat([x_2, x_1], 1)
