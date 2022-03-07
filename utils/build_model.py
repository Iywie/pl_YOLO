from lightning.yolox import LitYOLOX
from lightning.yolov3 import LitYOLOv3
from lightning.yolov5 import LitYOLOv5
from lightning.yolof import LitYOLOF
from lightning.resnet18 import LitResnet


CONFIGS = {
    # detection
    'yolox': LitYOLOX,
    'yolov3': LitYOLOv3,
    'yolov5': LitYOLOv5,
    'yolof': LitYOLOF,

    # classfication
    'resnet': LitResnet,
}


def build_model(model_name):
    return CONFIGS[model_name]
