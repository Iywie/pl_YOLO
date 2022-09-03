import torch.nn as nn
# backbones
from models.backbones.darknet_csp import CSPDarkNet
from models.backbones.mobilenext_csp import CSPMobileNext
from models.backbones.eelan import EELAN
from models.backbones.darknet_new import NewNet
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


def build_model(cfg_models):
    cb = cfg_models['backbone']
    cn = cfg_models['neck']
    ch = cfg_models['head']
    cl = cfg_models['loss']

    backbone = eval(cb['name'])(cb)
    neck = eval(cn['name'])(cn)
    head = eval(ch['name'])(ch)
    loss = eval(cl['name'])(cl)
    model = OneStageD(backbone, neck, head, loss)
    return model


class OneStageD(nn.Module):

    def __init__(self, backbone=None, neck=None, head=None, loss=None):
        super().__init__()
        self.backbone = backbone
        self.neck = neck
        self.head = head
        self.loss = loss

    def forward(self, x, labels):
        x = self.backbone(x)
        if self.neck is not None:
            x = self.neck(x)
        x = self.head(x)
        output = self.loss(x, labels)
        return output


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


def newnet(cfg):
    backbone = NewNet(cfg['depths'], cfg['channels'], cfg['outputs'], cfg['norm'], cfg['act'])
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


# Heads
def decoupled_head(cfg):
    head = DecoupledHead(cfg['num_class'], cfg['num_anchor'], cfg['channels'], cfg['norm'], cfg['act'])
    return head


def implicit_head(cfg):
    head = ImplicitHead(cfg['num_class'], cfg['num_anchor'], cfg['channels'])
    return head


# Losses
def yolox(cfg):
    head = YOLOXLoss(cfg['num_class'], cfg['stride'])
    return head


def yolov7(cfg):
    head = YOLOv7Loss(cfg['num_class'], cfg['stride'], cfg['anchors'])
    return head
