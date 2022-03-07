import torch
import torch.nn as nn
from models.layers.activation import get_activation
from models.layers.normalization import get_normalization


class DilatedEncoder(nn.Module):
    """
    Dilated Encoder for YOLOF.

    This module contains two types of components:
        - the original FPN lateral convolution layer and fpn convolution layer,
          which are 1x1 conv + 3x3 conv
        - the dilated residual block
    """

    def __init__(self,
                 in_channels=(256, 512, 1024),
                 encoder_channels=512,
                 mid_channels=128,
                 num_blocks=8,
                 block_dilations=None,
                 norm='bn',
                 act="silu",
                 ):
        super(DilatedEncoder, self).__init__()
        if block_dilations is None:
            block_dilations = [1, 2, 3, 4, 5, 6, 7, 8]
        self.in_channels = in_channels[-1]
        self.encoder_channels = encoder_channels
        self.block_mid_channels = mid_channels
        self.num_residual_blocks = num_blocks
        self.block_dilations = block_dilations

        self.lateral_conv = nn.Conv2d(self.in_channels,
                                      self.encoder_channels,
                                      kernel_size=(1, 1))
        self.lateral_norm = get_normalization(norm, self.encoder_channels)
        self.fpn_conv = nn.Conv2d(self.encoder_channels,
                                  self.encoder_channels,
                                  kernel_size=(3, 3),
                                  padding=1)
        self.fpn_norm = get_normalization(norm, self.encoder_channels)

        encoder_blocks = []
        for i in range(self.num_residual_blocks):
            dilation = self.block_dilations[i]
            encoder_blocks.append(
                Bottleneck(
                    self.encoder_channels,
                    self.block_mid_channels,
                    dilation=dilation,
                    norm_type=norm,
                    act_type=act
                )
            )
        self.dilated_encoder_blocks = nn.Sequential(*encoder_blocks)

    def forward(self, feature: torch.Tensor) -> list:
        out = self.lateral_norm(self.lateral_conv(feature))
        out = self.fpn_norm(self.fpn_conv(out))
        out = self.dilated_encoder_blocks(out)
        return [out]


class Bottleneck(nn.Module):

    def __init__(self,
                 in_channels: int = 512,
                 mid_channels: int = 128,
                 dilation: int = 1,
                 norm_type: str = 'BN',
                 act_type: str = 'ReLU'):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=(1, 1), padding=0),
            get_normalization(norm_type, mid_channels),
            get_activation(act_type)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(mid_channels, mid_channels,
                      kernel_size=(3, 3), padding=dilation, dilation=dilation),
            get_normalization(norm_type, mid_channels),
            get_activation(act_type)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(mid_channels, in_channels, kernel_size=(1, 1), padding=0),
            get_normalization(norm_type, in_channels),
            get_activation(act_type)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = out + identity
        return out
