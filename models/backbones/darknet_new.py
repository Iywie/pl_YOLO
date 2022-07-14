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
from torchvision.ops import deform_conv2d
from models.layers.network_blocks import Focus, BaseConv, SPPBottleneck
from models.layers.activation import get_activation
from models.layers.normalization import get_normalization
from models.layers.attention import SELayer, SALayer, SKFF, ECALayer, GAMLayer, ChannelAttention, MultiSpectralAttentionLayer, SimAM
from models.layers.swin_transformer import SwinTransformerLayer


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

        expand = 0.5
        attention = None
        # stage1
        self.stage1 = nn.Sequential(
            BaseConv(channels[0], channels[1], 3, 2, norm=norm, act="relu"),
            CSPLayer(channels[1], num_bottle=depths[0], expansion=expand, norm=norm, act=act, attn=attention),
            nn.ReLU()
        )

        expand = 0.5
        attention = None
        # stage2
        self.stage2 = nn.Sequential(
            BaseConv(channels[1], channels[2], 3, 2, norm=norm, act="relu"),
            CSPLayer(channels[2], num_bottle=depths[1], expansion=expand, norm=norm, act="hswish", attn=attention),
            nn.ReLU()
        )

        expand = 0.5
        attention = None
        # stage3
        self.stage3 = nn.Sequential(
            BaseConv(channels[2], channels[3], 3, 2, norm=norm, act="relu"),
            CSPLayer(channels[3], num_bottle=depths[2], expansion=expand, norm=norm, act="hswish", attn=attention),
            nn.ReLU()
        )

        expand = 0.5
        attention = None
        # stage4
        self.stage4 = nn.Sequential(
            BaseConv(channels[3], channels[4], 3, 2, norm=norm, act="relu"),
            SPPBottleneck(channels[4], channels[4], norm=norm, act="hswish"),
            CSPLayer(channels[4], num_bottle=depths[3], expansion=expand, shortcut=False, norm=norm, act="hswish", attn=attention),
            nn.ReLU()
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
        in_channels,
        num_bottle=1,
        expansion=1,
        shortcut=True,
        norm='bn',
        act="silu",
        attn=None,
    ):
        """
        Args:
            in_channels (int): input channels.
            out_channels (int): output channels.
            num_bottle (int): number of Bottlenecks. Default value: 1.
            shortcut (bool): residual operation.
            expansion (float): the number that hidden channels compared with output channels.
            norm (str): type of normalization
            act (str): type of activation
        """
        super().__init__()
        bottle_attn = None
        in_ch = in_channels // 2
        hidden_channels = int(in_ch * expansion)  # hidden channels
        self.conv1 = BaseConv(in_channels, in_ch, 1, stride=1, norm=norm, act=act)
        self.conv2 = BaseConv(in_channels, in_ch, 1, stride=1, norm=norm, act=act)
        self.m = nn.Sequential(
            *[Bottleneck(in_ch, in_ch, 1, shortcut, 2.0, norm=norm, act=act, attn=bottle_attn)
            for _ in range(num_bottle-1)]
        )
        self.use_attn = False
        if attn is not None:
            self.use_attn = True
            self.attn = attn
        # self.conv3 = BaseConv(in_channels, in_channels, 1, stride=1, norm=norm, act=act)

    def forward(self, x):
        x_1 = self.conv1(x)
        x_2 = self.conv2(x)
        # x_2, x_1 = channel_shuffle(x)
        x_1 = self.m(x_1)
        x = torch.cat((x_2, x_1), dim=1)
        # x = self.conv3(x)
        if self.use_attn is True:
            x = self.attn(x)
        return x


class Bottleneck(nn.Module):
    # Standard bottleneck from ResNet
    def __init__(
        self,
        in_channels,
        out_channels,
        stride=1,
        shortcut=True,
        expansion=0.5,
        norm='bn',
        act="silu",
        attn=None,
    ):
        super().__init__()
        in_ch = in_channels
        out_ch = out_channels
        hidden_channels = int(out_ch * expansion)
        self.conv0 = BaseConv(in_ch, in_ch, 3, stride=stride, groups=in_ch, norm=norm, act=act)
        self.conv1 = BaseConv(in_ch, hidden_channels, 1, stride=1, norm=norm, act=act)
        self.conv2 = BaseConv(hidden_channels, out_ch, 1, stride=1, norm=norm, act=act)
        self.conv3 = BaseConv(out_ch, out_ch, 3, stride=stride, groups=out_ch, norm=norm, act=None)

        self.use_add = shortcut and in_channels == out_channels
        self.use_attn = False
        if attn is not None:
            self.use_attn = True
            self.attn = attn

    def forward(self, x):
        y = x
        y = self.conv0(y)
        y = self.conv1(y)
        y = self.conv2(y)
        y = self.conv3(y)
        if self.use_attn is True:
            y = self.attn(y)
        if self.use_add:
            y = y + x
        return y


def channel_shuffle(x):
    batchsize, num_channels, height, width = x.data.size()
    assert (num_channels % 4 == 0)
    x = x.reshape(batchsize * num_channels // 2, 2, height * width)
    x = x.permute(1, 0, 2)
    x = x.reshape(2, -1, num_channels // 2, height, width)
    return x[0], x[1]
