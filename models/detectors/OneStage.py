import torch.nn as nn


class OneStageD(nn.Module):
    """
    YOLOX model module. The module list is defined by create_yolov3_modules function.
    The network returns loss values from three YOLO layers during training
    and detection results during test.
    """

    def __init__(self, backbone=None, neck=None, head=None):
        super().__init__()
        self.backbone = backbone
        self.neck = neck
        self.head = head

    def forward(self, x):
        x = self.backbone(x)
        if self.neck is not None:
            x = self.neck(x)
        x = self.head(x)
        return x
