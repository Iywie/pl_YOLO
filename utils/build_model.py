from PL_Modules.yolox import LitYOLOX
from PL_Modules.yolov3 import LitYOLOv3
from PL_Modules.yolov5 import LitYOLOv5


CONFIGS = {
    # detection
    'yolox': LitYOLOX,
    'yolov3': LitYOLOv3,
    'yolov5': LitYOLOv5,
}


def build_model(model_name):
    return CONFIGS[model_name]
