import torch
import torch.nn as nn
from .activation import get_activation
from .normalization import get_normalization


class BaseConv(nn.Module):
    """A Convolution2d -> Normalization -> Activation"""
    def __init__(
        self, in_channels, out_channels, ksize, stride, padding=None, groups=1, bias=False, norm="bn", act="silu"
    ):
        super().__init__()
        # same padding
        if padding is None:
            pad = (ksize - 1) // 2
        else:
            pad = padding
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=ksize,
            stride=stride,
            padding=pad,
            groups=groups,
            bias=bias,
        )
        self.norm = get_normalization(norm, out_channels)
        self.act = get_activation(act, inplace=True)

    def forward(self, x):
        if self.norm is None and self.act is None:
            return self.conv(x)
        elif self.act is None:
            return self.norm(self.conv(x))
        elif self.norm is None:
            return self.act(self.conv(x))
        return self.act(self.norm(self.conv(x)))

    def fuseforward(self, x):
        return self.act(self.conv(x))


class Focus(nn.Module):
    """Focus width and height information into channel space."""

    def __init__(self, in_channels, out_channels, ksize=1, stride=1, norm='bn', act="silu"):
        super().__init__()
        self.conv = BaseConv(in_channels * 4, out_channels, ksize, stride, norm=norm, act=act)

    def forward(self, x):
        # shape of x (b,c,w,h) -> y(b,4c,w/2,h/2)
        patch_top_left = x[..., ::2, ::2]
        patch_top_right = x[..., ::2, 1::2]
        patch_bot_left = x[..., 1::2, ::2]
        patch_bot_right = x[..., 1::2, 1::2]
        x = torch.cat(
            (
                patch_top_left,
                patch_bot_left,
                patch_top_right,
                patch_bot_right,
            ),
            dim=1,
        )
        return self.conv(x)


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
        self.conv2 = BaseConv(hidden_channels, out_channels, 3, stride=1, norm=norm, act=act)
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


class SPPBottleneck(nn.Module):
    """Spatial pyramid pooling layer used in YOLOv3-SPP"""
    def __init__(
        self, in_channels, out_channels, kernel_sizes=(5, 9, 13), norm='bn', act="silu"
    ):
        super().__init__()
        hidden_channels = in_channels // 2
        self.conv1 = BaseConv(in_channels, hidden_channels, 1, stride=1, norm=norm, act=act)
        self.m = nn.ModuleList(
            [
                nn.MaxPool2d(kernel_size=ks, stride=1, padding=ks // 2)
                for ks in kernel_sizes
            ]
        )
        conv2_channels = hidden_channels * (len(kernel_sizes) + 1)
        self.conv2 = BaseConv(conv2_channels, out_channels, 1, stride=1, act=act)

    def forward(self, x):
        x = self.conv1(x)
        x = torch.cat([x] + [m(x) for m in self.m], dim=1)
        x = self.conv2(x)
        return x


class SPPCSPC(nn.Module):
    # CSP https://github.com/WongKinYiu/CrossStagePartialNetworks
    def __init__(self, c1, c2, k=(5, 9, 13)):
        super(SPPCSPC, self).__init__()
        self.cv1 = BaseConv(c1, c2, 1, 1)
        self.cv2 = BaseConv(c1, c2, 1, 1)
        self.cv3 = BaseConv(c2, c2, 3, 1)
        self.cv4 = BaseConv(c2, c2, 1, 1)
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])
        self.cv5 = BaseConv(4 * c2, c2, 1, 1)
        self.cv6 = BaseConv(c2, c2, 3, 1)
        self.cv7 = BaseConv(2 * c2, c2, 1, 1)

    def forward(self, x):
        x1 = self.cv4(self.cv3(self.cv1(x)))
        y1 = self.cv6(self.cv5(torch.cat([x1] + [m(x1) for m in self.m], 1)))
        y2 = self.cv2(x)
        return self.cv7(torch.cat((y1, y2), dim=1))
