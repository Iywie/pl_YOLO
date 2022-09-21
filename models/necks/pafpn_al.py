import torch
import torch.nn as nn
from models.layers.activation import get_activation
from models.layers.network_blocks import BaseConv


class AL_PAFPN(nn.Module):
    """
    Only proceed 3 layer input. Like stage2, stage3, stage4.
    """

    def __init__(
            self,
            depths=(1, 1, 1, 1),
            in_channels=(256, 512, 1024),
            norm='bn',
            act="silu",
    ):
        super().__init__()

        self.shrink_conv1 = BaseConv(in_channels[2], in_channels[1], 1, 1, norm=norm, act=act)
        self.shrink_conv2 = BaseConv(in_channels[2], in_channels[1], 1, 1, norm=norm, act=act)
        self.shrink_conv3 = BaseConv(in_channels[1], in_channels[0], 1, 1, norm=norm, act=act)
        self.shrink_conv4 = BaseConv(in_channels[1], in_channels[0], 1, 1, norm=norm, act=act)
        self.upsample = nn.Upsample(scale_factor=2, mode="bicubic")

        self.p5_p4 = CSPLayer(
            in_channels[1],
            num_bottle=depths[0],
            shortcut=False,
            norm=norm,
            act=act,
        )

        self.p4_p3 = CSPLayer(
            in_channels[0],
            num_bottle=depths[0],
            shortcut=False,
            norm=norm,
            act=act,
        )

        # bottom-up conv
        self.downsample_conv1 = BaseConv(int(in_channels[0]), int(in_channels[0]), 3, 2, norm=norm, act=act)
        self.downsample_conv2 = BaseConv(int(in_channels[1]), int(in_channels[1]), 3, 2, norm=norm, act=act)

        self.n3_n4 = CSPLayer(
            in_channels[1],
            num_bottle=depths[0],
            shortcut=False,
            norm=norm,
            act=act,
        )
        self.n4_n5 = CSPLayer(
            in_channels[2],
            num_bottle=depths[0],
            shortcut=False,
            norm=norm,
            act=act,
        )

    def forward(self, inputs):
        #  backbone
        [c3, c4, c5] = inputs
        # top-down
        p5 = c5
        p5_expand = self.shrink_conv1(p5)
        p5_upsample = self.upsample(p5_expand)
        p4 = torch.cat([p5_upsample, c4], 1)
        p4 = self.shrink_conv2(p4)
        p4 = self.p5_p4(p4)

        p4_expand = self.shrink_conv3(p4)
        p4_upsample = self.upsample(p4_expand)
        p3 = torch.cat([p4_upsample, c3], 1)
        p3 = self.shrink_conv4(p3)
        p3 = self.p4_p3(p3)

        # down-top
        n3 = p3
        n3_downsample = self.downsample_conv1(n3)
        n4 = torch.cat([n3_downsample, p4_expand], 1)
        n4 = self.n3_n4(n4)

        n4_downsample = self.downsample_conv2(n4)
        n5 = torch.cat([n4_downsample, p5_expand], 1)
        n5 = self.n4_n5(n5)

        outputs = (n3, n4, n5)
        return outputs


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
            num_bottle (int): number of Bottlenecks. Default value: 1.
            shortcut (bool): residual operation.
            expansion (float): the number that hidden channels compared with output channels.
            norm (str): type of normalization
            act (str): type of activation
        """
        super().__init__()
        in_ch = in_channels // 4
        num_conv = num_bottle // 2 if num_bottle > 2 else 1

        self.conv1 = BaseConv(in_channels, in_ch, 1, stride=1, norm=norm, act=act)
        self.conv2 = BaseConv(in_channels, in_ch, 1, stride=1, norm=norm, act=act)

        self.conv3 = nn.Sequential(
            *[Bottleneck(in_ch, in_ch, stride=1, shortcut=True, expansion=2, norm=norm, act=act)
              for _ in range(num_conv)]
        )
        self.conv4 = nn.Sequential(
            *[Bottleneck(in_ch, in_ch, stride=1, shortcut=True, expansion=2, norm=norm, act=act)
              for _ in range(num_conv)]
        )
        self.nonlinearity = get_activation(act)
        self.use_attn = False
        if attn is not None:
            self.use_attn = True
            self.attn = attn

    def forward(self, x):
        x_1 = self.conv1(x)
        x_2 = self.conv2(x)
        x_3 = self.conv3(x_2)
        x_4 = self.conv4(x_3)
        x_all = [x_1, x_2, x_3, x_4]
        x = torch.cat(x_all, dim=1)
        return x


class Bottleneck(nn.Module):
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
        self.conv0 = BaseConv(in_ch, in_ch, 3, stride=stride, groups=in_ch, norm=norm, act=None)
        self.conv1 = BaseConv(in_ch, hidden_channels, 1, stride=1, norm=None, act=act)
        self.conv2 = BaseConv(hidden_channels, out_ch, 1, stride=1, norm=norm, act=None)
        self.conv3 = BaseConv(out_ch, out_ch, 3, stride=stride, groups=out_ch, norm=norm, act=None)
        self.nonlinearity = get_activation(act)
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
        y = self.nonlinearity(y)
        return y
