"""
Deep Residual Learning for Image Recognition https://arxiv.org/abs/1512.03385v1
Depths:
    resnet18: BasicBlock, [2, 2, 2, 2]
    resnet34: BasicBlock, [3, 4, 6, 3]
    resnet50: BottleNeck, [3, 4, 6, 3]
    resnet101: BottleNeck, [3, 4, 23, 3]
    resnet152: BottleNeck, [3, 8, 36, 3]
"""

import torch.nn as nn
from torchvision.models import resnet


class ResNet(nn.Module):

    def __init__(
            self, 
            block,
            depths=(3, 4, 6, 3),
            channels=(64, 128, 256, 512),
            out_features=("stage2", "stage3", "stage4")
    ):
        super().__init__()

        if block == "BasicBlock":
            block = BasicBlock
        elif block == "BottleNeck":
            block = BottleNeck
        self.out_features = out_features
        self.in_channels = channels[0]

        self.stem = nn.Sequential(
            nn.Conv2d(3, channels[0], kernel_size=7, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(channels[0]),
            nn.ReLU(inplace=True)
        )

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.stage1 = self._make_layer(block, channels[0], depths[0], 1)
        self.stage2 = self._make_layer(block, channels[1], depths[1], 2)
        self.stage3 = self._make_layer(block, channels[2], depths[2], 2)
        self.stage4 = self._make_layer(block, channels[3], depths[3], 2)

    def _make_layer(self, block, out_channels, depth, stride):
        strides = [stride] + [1] * (depth - 1)  # [2, 1, 1, ...]
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        outputs = {}
        x = self.stem(x)
        outputs["stem"] = x
        x = self.maxpool(x)
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


class BasicBlock(nn.Module):
    """
    Basic Block for resnet 18 and resnet 34
    """
    # BasicBlock and BottleNeck block have different output size
    # we use class attribute expansion to distinct
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        # residual function
        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels * BasicBlock.expansion, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels * BasicBlock.expansion)
        )
        # shortcut
        self.shortcut = nn.Sequential()
        # the shortcut output dimension is not the same with residual function
        # use 1*1 convolution to match the dimension
        if stride != 1 or in_channels != BasicBlock.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * BasicBlock.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * BasicBlock.expansion)
            )

    def forward(self, x):
        return nn.ReLU(inplace=True)(self.residual_function(x) + self.shortcut(x))


class BottleNeck(nn.Module):
    """
    Residual block for resnet over 50 layers
    """
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, stride=stride, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels * BottleNeck.expansion, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels * BottleNeck.expansion),
        )

        self.shortcut = nn.Sequential()

        if stride != 1 or in_channels != out_channels * BottleNeck.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * BottleNeck.expansion, stride=stride, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels * BottleNeck.expansion)
            )

    def forward(self, x):
        return nn.ReLU(inplace=True)(self.residual_function(x) + self.shortcut(x))
