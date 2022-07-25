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
from models.layers.network_blocks import Focus, BaseConv, SPPBottleneck
from models.layers.activation import get_activation
from models.layers.normalization import get_normalization


class NewCSPDarkNet(nn.Module):
    """
    CSPDarkNet consists of five block: stem, dark2, dark3, dark4 and dark5.
    """
    def __init__(
        self,
        depths=(3, 9, 9, 3),
        channels=(64, 128, 256, 512, 1024),
        out_features=("stage2", "stage3", "stage4"),
        norm='bn',
        act="silu",
    ):
        super().__init__()

        # parameters of the network
        assert out_features, "please provide output features of Darknet!"
        self.out_features = out_features

        # stem
        self.stem = Focus(3, channels[0], ksize=3, norm=norm, act=act)

        # stage1
        self.stage1 = nn.Sequential(
            BaseConv(channels[0], channels[1], 3, 2, norm=norm, act=act),
            CSPLayer(channels[1], channels[1], num_bottle=depths[0], norm=norm, act=act),
        )

        # stage2
        self.stage2 = nn.Sequential(
            BaseConv(channels[1], channels[2], 3, 2, norm=norm, act=act),
            CSPLayer(channels[2], channels[2], num_bottle=depths[1], norm=norm, act=act),
        )

        # stage3
        self.stage3 = nn.Sequential(
            BaseConv(channels[2], channels[3], 3, 2, norm=norm, act=act),
            CSPLayer(channels[3], channels[3], num_bottle=depths[2], norm=norm, act=act),
        )

        # stage4
        self.stage4 = nn.Sequential(
            BaseConv(channels[3], channels[4], 3, 2, norm=norm, act=act),
            SPPBottleneck(channels[4], channels[4], norm=norm, act=act),
            CSPLayer(channels[4], channels[4], num_bottle=depths[3], shortcut=False, norm=norm, act=act),
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


class Bottleneck(nn.Module):
    # Standard bottleneck from ResNet
    def __init__(
        self,
        in_channels,
        out_channels,
        shortcut=True,
        expansion=0.5,
        norm='bn',
        act="silu",
    ):
        super().__init__()
        hidden_channels = int(out_channels * expansion)
        self.bn = get_normalization(norm, out_channels)
        self.act = get_activation(act, inplace=True)
        self.conv1 = BaseConv(in_channels, hidden_channels, 1, stride=1, norm=norm, act=act)
        self.conv2 = BaseConv(hidden_channels, out_channels, 3, stride=1, groups=hidden_channels, norm=norm, act=act)
        self.use_add = shortcut and in_channels == out_channels

    def forward(self, x):
        y = self.conv2(self.conv1(x))
        if self.use_add:
            y = y + x
        return y


class CSPLayer(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        num_bottle=1,
        shortcut=True,
        expansion=0.5,
        norm='bn',
        act="silu",
    ):
        """
        Args:
            in_channels (int): input channels.
            out_channels (int): output channels.
            num_bottle (int): number of Bottlenecks. Default value: 1.
            shortcut (bool): residual operation.
            expansion (int): the number that hidden channels compared with output channels.
            norm (str): type of normalization
            act (str): type of activation
        """
        super().__init__()
        hidden_channels = int(out_channels * expansion)  # hidden channels
        self.conv1 = BaseConv(in_channels, hidden_channels, 1, stride=1, norm=norm, act=act)
        self.conv2 = BaseConv(in_channels, hidden_channels, 1, stride=1, norm=norm, act=act)
        self.conv3 = BaseConv(2 * hidden_channels, out_channels, 1, stride=1, norm=norm, act=act)
        module_list = [
            Bottleneck(hidden_channels, hidden_channels, shortcut, 1.0, norm=norm, act=act)
            for _ in range(num_bottle)
        ]
        self.m = nn.Sequential(*module_list)

    def forward(self, x):
        x_1 = self.conv1(x)
        x_2 = self.conv2(x)
        x_1 = self.m(x_1)
        x = torch.cat((x_1, x_2), dim=1)
        return self.conv3(x)