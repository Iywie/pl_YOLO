import torch
import torch.nn as nn

from models.layers.activation import get_activation
from models.layers.normalization import get_normalization
from models.layers.network_blocks import BaseConv
from models.backbones.swinv2 import Mlp, window_partition, window_reverse, WindowAttention
from models.layers.drops import DropPath
from models.layers.weight_init import trunc_normal_
from models.utils.helpers import to_2tuple


class NewPAFPN(nn.Module):
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

        # self.p5_p4 = SwinTransformerLayer(
        #     in_channels[1], input_resolution=(14, 14), num_heads=1, window_size=7, shift_size=0,
        # )
        self.p5_p4 = CSPLayer(
            in_channels[1],
            num_bottle=depths[0],
            shortcut=False,
            norm=norm,
            act=act,
        )
        # self.p4_p3 = SwinTransformerLayer(
        #     in_channels[0], input_resolution=(28, 28), num_heads=1, window_size=7, shift_size=3,
        # )

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
        # self.n3_n4 = SwinTransformerLayer(
        #     in_channels[1], input_resolution=(14, 14), num_heads=1, window_size=7, shift_size=0,
        # )
        # self.n4_n5 = SwinTransformerLayer(
        #     in_channels[2], input_resolution=(7, 7), num_heads=1, window_size=7, shift_size=0,
        # )

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
        bottle_attn = None
        in_ch = in_channels // 2
        hidden_channels = int(in_ch * expansion)  # hidden channels
        self.conv1 = BaseConv(in_channels, in_ch, 1, stride=1, norm=norm, act=act)
        self.conv2 = BaseConv(in_channels, in_ch, 1, stride=1, norm=norm, act=act)
        self.m = nn.Sequential(
            *[Bottleneck(in_ch, in_ch, 1, shortcut, 2, norm=norm, act=act, attn=bottle_attn)
              for _ in range(num_bottle - 1)]
        )
        self.nonlinearity = get_activation(act)
        self.use_attn = False
        if attn is not None:
            self.use_attn = True
            self.attn = attn

    def forward(self, x):
        x_1 = self.conv1(x)
        x_2 = self.conv2(x)
        x_1 = self.m(x_1)
        x = torch.cat((x_2, x_1), dim=1)
        if self.use_attn is True:
            x = self.attn(x)
        x = self.nonlinearity(x)
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


class SwinTransformerLayer(nn.Module):
    def __init__(
            self, dim, input_resolution, num_heads, window_size=7, shift_size=0,
            mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0., drop_path=0.,
            act_layer=nn.GELU, norm_layer=nn.LayerNorm, pretrained_window_size=0
    ):
        super().__init__()
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        self.input_resolution = input_resolution
        if min(self.input_resolution) <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop,
            pretrained_window_size=to_2tuple(pretrained_window_size))

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        if self.shift_size > 0:
            # calculate attention mask for SW-MSA
            H, W = self.input_resolution
            img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
            h_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1

            mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)  # B h*w C
        B, L, C = x.shape

        shortcut = x
        x = x.view(B, H, W, C)
        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask=self.attn_mask)  # nW*B, window_size*window_size, C

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        x = x.view(B, H * W, C)
        x = shortcut + self.drop_path(self.norm1(x))

        # FFN
        x = x + self.drop_path(self.norm2(self.mlp(x)))

        x = x.view(B, H, W, C).permute(0, 3, 1, 2).contiguous()

        return x
