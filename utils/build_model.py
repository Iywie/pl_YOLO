from PL_Modules.yolox import LitYOLOX
from PL_Modules.yolov3 import LitYOLOv3
from PL_Modules.yolov5 import LitYOLOv5
from PL_Modules.yolox_dw import LitDWYOLOX
#  BACKBONE
from models.backbones.darknet_csp import CSPDarkNet
from models.backbones.resnet import ResNet
from models.backbones.convnext import ConvNeXt
from models.backbones.shufflenetv2 import ShuffleNetV2_Plus
from models.backbones.mobilenetv3 import MobileNetV3_Large
from models.backbones.swinv2 import SwinTransformerV2
from models.backbones.ghostnet import GhostNet
from models.backbones.mobilenext import MobileNeXt
from models.backbones.darknet_new import NewCSPDarkNet
# NECK
from models.necks.pafpn_csp import CSPPAFPN
from models.necks.pafpn_new import NewPAFPN
# HEAD
from models.heads.decoupled_head import DecoupledHead
from models.heads.yolor.yolor_decoupled_head import YOLORDecoupledHead
from models.heads.pp_yoloe.ppyoloe_decoupled_head import PPYOLOEDecoupledHead
from models.heads.yolox.yolox_sa_head import YOLOXSADecoupledHead
# LOSS
from models.heads.yolox.yolox_loss import YOLOXLoss


CONFIGS = {
    # detection
    'yolox': LitYOLOX,
    'yolov3': LitYOLOv3,
    'yolov5': LitYOLOv5,
    'dwyolox': LitDWYOLOX,
}

def build_model(model_name):
    return CONFIGS[model_name]


BACKBONE = {
    'resnet': ResNet,
    'cspdarknet': CSPDarkNet,
    'convnext': ConvNeXt,
    'shufflenetv2': ShuffleNetV2_Plus,
    'mobilenetv3': MobileNetV3_Large,
    'ghostnet': GhostNet,
    'mobilenext': MobileNeXt,
    'swintransformerv2': SwinTransformerV2,
    'mycspdarknet': NewCSPDarkNet
}

NECK = {
    'csppafpn': CSPPAFPN,
    'mypafpn': NewPAFPN,
}

HEAD = {
    'decoupledhead': DecoupledHead,
    'mydecoupledhead': YOLOXSADecoupledHead,
}

LOSS = {
    'yoloxloss': YOLOXLoss
}


def build(configs):
    backbone = BACKBONE[configs['backbone']]
    neck = NECK[configs['backbone']]
    head = HEAD[configs['backbone']]
