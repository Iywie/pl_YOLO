import torch
import torch.nn as nn
import torch.nn.functional as F
from models.layers.network_blocks import BaseConv


class RPNHead(nn.Module):
    """RPN head.
        Args:
            in_channels (list): Number of channels in the input feature map.
            num_convs (int): Number of convolution layers in the head. Default 1.
    """
    def __init__(self,
                 num_classes,
                 n_anchors,
                 in_channels,
                 feat_channels,
                 num_convs=1,
                 norm='bn',
                 act='relu',
                 ):
        super().__init__()
        self.rpn_conv = nn.ModuleList()
        for i in range(len(in_channels)):
            if num_convs > 1:
                rpn_convs = []
                for j in range(num_convs):
                    if i == 0:
                        in_c = in_channels[i]
                    else:
                        in_c = feat_channels
                    rpn_convs.append(
                        BaseConv(in_c, feat_channels, ksize=3, stride=1, norm=norm, act=act)
                    )
                self.rpn_conv.append(nn.Sequential(*rpn_convs))
            else:
                self.rpn_conv.append(nn.Conv2d(in_channels[i], feat_channels, 3, padding=1))
        self.rpn_cls = nn.Conv2d(feat_channels, n_anchors, 1)
        self.rpn_reg = nn.Conv2d(feat_channels, n_anchors * 4, 1)

    def forward(self, inputs):
        outputs = []
        for k, (rpn_conv, x) in enumerate(zip(self.rpn_conv, inputs)):
            x = rpn_conv(x)
            x = F.relu(x, inplace=True)
            rpn_cls_score = self.rpn_cls(x)
            rpn_bbox_pred = self.rpn_reg(x)
            # output: [batch_size, n_ch, h, w]
            output = torch.cat([rpn_bbox_pred, rpn_cls_score], 1)
            outputs.append(output)
        return outputs
