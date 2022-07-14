"""
Very Deep Convolutional Networks for Large-Scale Image Recognition.
https://arxiv.org/abs/1409.1556v6
Depths:
    VGG-A: [1, 1, 2, 2, 2]
    VGG-B: [2, 2, 2, 2, 2]
    VGG-D: [2, 2, 3, 3, 3]
    VGG-E: [2, 2, 4, 4, 4]
"""
import torch.nn as nn


class VGG(nn.Module):

    def __init__(self, num_blocks, channels):
        super().__init__()

        self.in_channels = 3
        self.batch_norm = False

        self.conv1 = self._make_layer(channels[0], num_blocks[0])
        self.conv2 = self._make_layer(channels[1], num_blocks[1])
        self.conv3 = self._make_layer(channels[2], num_blocks[2])
        self.conv4 = self._make_layer(channels[3], num_blocks[3])
        self.conv5 = self._make_layer(channels[4], num_blocks[4])

    def _make_layer(self, out_channels, num_block):
        layers = []
        for i in range(num_block):
            layers.append(nn.Conv2d(self.in_channels, out_channels, kernel_size=3, padding=1))
            self.in_channels = out_channels
            if self.batch_norm:
                layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        return nn.Sequential(*layers)

    def forward(self, x):
        output = self.conv1(x)
        output = self.conv2(output)
        output = self.conv3(output)
        output = self.conv4(output)
        output = self.conv5(output)
        return output
