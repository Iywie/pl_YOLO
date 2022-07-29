"""
Channel
    base: [16, 96, 192, 384, 1280]
    small: [16, 48, 96, 192, 1280]
"""

import torch.nn as nn
import math

__all__ = ['mobilenext']


class MobileNext(nn.Module):
    def __init__(
            self,
            channels=(16, 32, 64, 128, 256),
            out_features=("stage2", "stage3", "stage4"),
    ):
        super().__init__()
        self.out_features = out_features
        # building first layer
        self.stem = nn.Sequential(
            nn.Conv2d(3, channels[0], 3, 2, 1, bias=False),
            nn.BatchNorm2d(channels[0]),
            nn.ReLU6(inplace=True)
        )
        # building inverted residual blocks
        block = SGBlock

        self.stage1 = nn.Sequential(
            block(channels[0], channels[1], 2, 2),
            block(channels[1], channels[1], 1, 6, True),
        )
        self.stage2 = nn.Sequential(
            block(channels[1], channels[2], 2, 6),
            block(channels[2], channels[2], 1, 6),
            block(channels[2], channels[2], 1, 6),
        )
        self.stage3 = nn.Sequential(
            block(channels[2], int(channels[2]*1.5), 2, 6),
            block(int(channels[2]*1.5), int(channels[2]*1.5), 1, 6),
            block(int(channels[2]*1.5), int(channels[2]*1.5), 1, 6),
            block(int(channels[2]*1.5), channels[3], 1, 6),
            block(channels[3], channels[3], 1, 6),
            block(channels[3], channels[3], 1, 6),
            block(channels[3], channels[3], 1, 6),
        )
        self.stage4 = nn.Sequential(
            block(channels[3], int(channels[3] * 1.5), 2, 6),
            block(int(channels[3] * 1.5), int(channels[3] * 1.5), 1, 6),
            block(int(channels[3] * 1.5), int(channels[3] * 1.5), 1, 6),
            block(int(channels[3] * 1.5), int(channels[3] * 1.5), 1, 6),
            block(int(channels[3] * 1.5), int(channels[3] * 2.5), 1, 6),
            block(int(channels[3] * 2.5), int(channels[3] * 2.5), 1, 6),
            block(int(channels[3] * 2.5), int(channels[3] * 2.5), 1, 6),
            block(int(channels[3] * 2.5), channels[4], 1, 6, True),
        )
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

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


class ConvBNReLU(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, groups=1):
        padding = (kernel_size - 1) // 2
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, groups=groups, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.ReLU6(inplace=True)
        )


def conv_3x3_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )


def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )


class SGBlock(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio, keep_3x3=False):
        super(SGBlock, self).__init__()
        assert stride in [1, 2]

        hidden_dim = inp // expand_ratio
        if hidden_dim < oup / 6.:
            hidden_dim = math.ceil(oup / 6.)
            hidden_dim = _make_divisible(hidden_dim, 16)  # + 16

        # self.relu = nn.ReLU6(inplace=True)
        self.identity = False
        self.identity_div = 1
        self.expand_ratio = expand_ratio
        if expand_ratio == 2:
            self.conv = nn.Sequential(
                # dw
                nn.Conv2d(inp, inp, 3, 1, 1, groups=inp, bias=False),
                nn.BatchNorm2d(inp),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU6(inplace=True),
                # dw
                nn.Conv2d(oup, oup, 3, stride, 1, groups=oup, bias=False),
                nn.BatchNorm2d(oup),
            )
        elif inp != oup and stride == 1 and keep_3x3 == False:
            self.conv = nn.Sequential(
                # pw-linear
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU6(inplace=True),
            )
        elif inp != oup and stride == 2 and keep_3x3 == False:
            self.conv = nn.Sequential(
                # pw-linear
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU6(inplace=True),
                # dw
                nn.Conv2d(oup, oup, 3, stride, 1, groups=oup, bias=False),
                nn.BatchNorm2d(oup),
            )
        else:
            if keep_3x3 == False:
                self.identity = True
            self.conv = nn.Sequential(
                # dw
                nn.Conv2d(inp, inp, 3, 1, 1, groups=inp, bias=False),
                nn.BatchNorm2d(inp),
                nn.ReLU6(inplace=True),
                # pw
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                # nn.ReLU6(inplace=True),
                # pw
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU6(inplace=True),
                # dw
                nn.Conv2d(oup, oup, 3, 1, 1, groups=oup, bias=False),
                nn.BatchNorm2d(oup),
            )

    def forward(self, x):
        out = self.conv(x)

        if self.identity:
            return out + x
        else:
            return out


def sg_block_extra(inp, oup, stride, expand_ratio, last=False):
    hidden_dim = int(inp * expand_ratio)
    conv = nn.Sequential(
        # dw
        nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
        nn.BatchNorm2d(inp),
        nn.ReLU6(inplace=True),
        # pw
        nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
        nn.BatchNorm2d(hidden_dim),
        # pw
        nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True),
    )
    return conv


def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v
