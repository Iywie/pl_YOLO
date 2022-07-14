import torch
import torch.nn as nn

from models.layers.network_blocks import BaseConv, CSPLayer


class CSPPAFPN(nn.Module):
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
        self.shrink_conv2 = BaseConv(in_channels[1], in_channels[0], 1, 1, norm=norm, act=act)
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
        self.p5_p4 = CSPLayer(
            2 * in_channels[1],
            in_channels[1],
            num_bottle=depths[0],
            shortcut=False,
            norm=norm,
            act=act,
        )
        self.p4_p3 = CSPLayer(
            2 * in_channels[0],
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
            2 * in_channels[0],
            in_channels[1],
            num_bottle=depths[0],
            shortcut=False,
            norm=norm,
            act=act,
        )
        self.n4_n5 = CSPLayer(
            2 * in_channels[1],
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
        p4 = self.p5_p4(p4)

        p4_expand = self.shrink_conv2(p4)
        p4_upsample = self.upsample(p4_expand)
        p3 = torch.cat([p4_upsample, c3], 1)
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
