import torch.nn as nn


class SimpleHead(nn.Module):
    def __init__(
            self,
            num_classes=80,
            n_anchors=1,
            in_channels=None,
    ):
        super().__init__()
        self.n_anchors = n_anchors
        self.num_classes = num_classes
        self.head = nn.ModuleList()
        # For each feature map we go through different convolution.
        for i in range(len(in_channels)):
            self.head.append(
                nn.Conv2d(in_channels[i], n_anchors * (5 + num_classes), 1)
            )

    def forward(self, inputs):
        outputs = []
        for k, (head_conv, x) in enumerate(zip(self.head, inputs)):
            # x: [batch_size, n_ch, h, w]
            output = head_conv[k](x)
            outputs.append(output)
        return outputs
