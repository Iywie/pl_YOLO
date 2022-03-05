import torch.nn as nn


class ClsHead(nn.Module):

    def __init__(self, channel, num_classes=100):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(channel, num_classes)

    def forward(self, x):
        output = self.avg_pool(x)
        output = output.view(output.size(0), -1)
        output = self.fc(output)
        return output
