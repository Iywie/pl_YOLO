import torch.nn as nn
# backbones
from models.backbones.darknet_csp import CSPDarkNet
from models.backbones.mobilenext_csp import CSPMobileNext
from models.backbones.eelan import EELAN
from models.backbones.ecmnet import ECMNet
from models.backbones.shufflenetv2 import ShuffleNetV2_Plus
from models.backbones.mobilenetv3 import MobileNetV3_Small, MobileNetV3_Large
from models.backbones.vision_transformer.vision_transformer import VisionTransformer
from models.backbones.vision_transformer.swin_transformer import SwinTransformer
# necks
from models.necks.pafpn_csp import CSPPAFPN
from models.necks.pafpn_al import AL_PAFPN
from models.necks.yolov7_neck import YOLOv7NECK
# heads
from models.heads.decoupled_head import DecoupledHead
from models.heads.implicit_head import ImplicitHead
# loss
from models.losses.yolox.yolox_loss import YOLOXLoss
from models.losses.yolov7.yolov7_loss import YOLOv7Loss


def build_model(cfg_models, num_classes):
    cb = cfg_models['backbone']
    cn = cfg_models['neck']
    ch = cfg_models['head']
    cl = cfg_models['loss']

    backbone = eval(cb['name'])(cb)
    neck = eval(cn['name'])(cn)
    head = eval(ch['name'])(ch, num_classes)
    loss = eval(cl['name'])(cl, num_classes)
    model = OneStageD(backbone, neck, head, loss)
    return model


class OneStageD(nn.Module):

    def __init__(self, backbone=None, neck=None, head=None, loss=None):
        super().__init__()
        self.backbone = backbone
        self.neck = neck
        self.head = head
        self.loss = loss

    def forward(self, x, labels=None):
        x = self.backbone(x)
        if self.neck is not None:
            x = self.neck(x)
        x = self.head(x)
        if labels is not None:
            x = self.loss(x, labels)
        return x


# Backbones
def cspdarknet(cfg):
    backbone = CSPDarkNet(cfg['depths'], cfg['channels'], cfg['outputs'], cfg['norm'], cfg['act'])
    return backbone


def cspmobilenext(cfg):
    backbone = CSPMobileNext(cfg['depths'], cfg['channels'], cfg['outputs'], cfg['norm'], cfg['act'])
    return backbone


def eelan(cfg):
    backbone = EELAN(cfg['depths'], cfg['channels'], cfg['outputs'], cfg['norm'], cfg['act'])
    return backbone


def ecmnet(cfg):
    backbone = ECMNet(cfg['depths'], cfg['channels'], cfg['outputs'], cfg['norm'], cfg['act'])
    return backbone


def shufflenetv2(cfg):
    backbone = ShuffleNetV2_Plus(cfg['channels'], cfg['outputs'], cfg['norm'], cfg['act'])
    return backbone


def mobilenetv3s(cfg):
    backbone = MobileNetV3_Small(cfg['outputs'])
    return backbone


def mobilenetv3l(cfg):
    backbone = MobileNetV3_Large(cfg['outputs'])
    return backbone


def vision_transformer(cfg):
    backbone = VisionTransformer(patch_size=cfg['patch_size'], embed_dim=cfg['embed_dim'], depth=cfg['depth'],
                                 num_heads=cfg['num_heads'], mlp_ratio=cfg['mlp_ratio'])
    return backbone


def swin_transformer(cfg):
    backbone = SwinTransformer(embed_dim=cfg['embed_dim'], depths=cfg['depths'], num_heads=cfg['num_heads'],
                               window_size=cfg['window_size'], mlp_ratio=cfg['mlp_ratio'],
                               drop_path_rate=cfg['drop_path_rate'])
    return backbone


# Necks
def csppafpn(cfg):
    neck = CSPPAFPN(cfg['depths'], cfg['channels'], cfg['norm'], cfg['act'])
    return neck


def al_pafpn(cfg):
    neck = AL_PAFPN(cfg['depths'], cfg['channels'], cfg['norm'], cfg['act'])
    return neck


def yolov7neck(cfg):
    neck = YOLOv7NECK(cfg['depths'], cfg['channels'], cfg['norm'], cfg['act'])
    return neck


def none(cfg):
    return None


# Heads
def decoupled_head(cfg, num_classes):
    head = DecoupledHead(num_classes, cfg['num_anchor'], cfg['channels'], cfg['norm'], cfg['act'])
    return head


def implicit_head(cfg, num_classes):
    head = ImplicitHead(num_classes, cfg['num_anchor'], cfg['channels'])
    return head


# Losses
def yolox(cfg, num_classes):
    head = YOLOXLoss(num_classes, cfg['stride'])
    return head


def yolov7(cfg, num_classes):
    head = YOLOv7Loss(num_classes, cfg['stride'], cfg['anchors'])
    return head
