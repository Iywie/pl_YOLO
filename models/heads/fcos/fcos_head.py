import torch
import torch.nn as nn
from models.layers.network_blocks import BaseConv


class DecoupledHead(nn.Module):
    def __init__(
            self,
            num_classes=80,
            n_anchors=1,
            in_channels=None,
            norm='bn',
            act="silu",
    ):
        super().__init__()
        self.n_anchors = n_anchors
        self.num_classes = num_classes
        ch = self.n_anchors * self.num_classes
        conv = BaseConv
        self.feat_channels = 256
        self.stems = nn.ModuleList()
        self.cls_convs = nn.ModuleList()
        self.cls_preds = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        self.reg_preds = nn.ModuleList()
        self.obj_preds = nn.ModuleList()

        self.conv_centerness = nn.Conv2d(self.feat_channels, 1, 3, padding=1)

        # For each feature map we go through different convolution.
        for i in range(len(in_channels)):

            self.cls_convs.append(
                nn.Sequential(
                    *[
                        conv(in_channels[i], self.feat_channels, ksize=3, stride=1, norm=norm, act=act),
                        conv(self.feat_channels, self.feat_channels, ksize=3, stride=1, norm=norm, act=act),
                    ]
                )
            )
            self.cls_preds.append(
                nn.Conv2d(self.feat_channels, ch, kernel_size=(1, 1), stride=(1, 1), padding=0)
            )

            self.reg_convs.append(
                nn.Sequential(
                    *[
                        conv(in_channels[i], self.feat_channels, ksize=3, stride=1, norm=norm, act=act),
                        conv(self.feat_channels, self.feat_channels, ksize=3, stride=1, norm=norm, act=act),
                    ]
                )
            )

            self.reg_preds.append(
                nn.Conv2d(self.feat_channels, self.n_anchors * 4, kernel_size=(1, 1), stride=(1, 1), padding=0)
            )
            self.obj_preds.append(
                nn.Conv2d(self.feat_channels, self.n_anchors * 1, kernel_size=(1, 1), stride=(1, 1), padding=0)
            )

    def forward(self, inputs):
        outputs = []
        for k, (cls_conv, reg_conv, x) in enumerate(zip(self.cls_convs, self.reg_convs, inputs)):
            # Change all inputs to the same channel.
            x = self.stems[k](x)

            cls_x = x
            reg_x = x

            cls_feat = cls_conv(cls_x)
            cls_output = self.cls_preds[k](cls_feat)
            reg_feat = reg_conv(reg_x)
            reg_output = self.reg_preds[k](reg_feat)
            obj_output = self.obj_preds[k](reg_feat)

            # output: [batch_size, n_ch, h, w]
            output = torch.cat([reg_output, obj_output, cls_output], 1)
            outputs.append(output)
        return outputs
