import torch.nn as nn
# backbones
from models.backbones.darknet_csp import CSPDarkNet
# necks
from models.necks.pafpn_csp import CSPPAFPN
# heads
from models.heads.yolox.decoupled_head import DecoupledHead
# loss
from models.heads.yolox.yolox_loss import YOLOXLoss
# decoder
from models.heads.yolox.yolox_decoder import YOLOXDecoder


def build_model(cfg_models):
    cb = cfg_models['backbone']
    cn = cfg_models['neck']
    ch = cfg_models['head']
    cl = cfg_models['loss']

    backbones = {
        'cspdarknet': CSPDarkNet(cb['depths'], cb['channels'], cb['outputs'], cb['norm'], cb['act']),
    }

    necks = {
        'csppafpn': CSPPAFPN(cn['depths'], cn['channels'], cn['norm'], cn['act']),
    }

    heads = {
        'decoupled_head': DecoupledHead(ch['num_class'], ch['num_anchor'], cn['channels'], ch['norm'], ch['act']),
    }

    losses = {
        'simOTA': YOLOXLoss(ch['num_class'], ch['stride'])
    }

    decoders = {
        'simOTA': YOLOXDecoder(ch['num_class'], ch['stride'])
    }

    backbone = backbones.get(cb['name'])
    neck = necks.get(cn['name'])
    head = heads.get(ch['name'])
    loss = losses.get(cl['name'])
    decoder = decoders.get(cl['name'])
    model = OneStageD(backbone, neck, head, loss, decoder)
    return model


class OneStageD(nn.Module):

    def __init__(self, backbone=None, neck=None, head=None, loss=None, decoder=None):
        super().__init__()
        self.backbone = backbone
        self.neck = neck
        self.head = head
        self.loss = loss
        self.decoder = decoder

    def forward(self, x, labels=None):
        x = self.backbone(x)
        if self.neck is not None:
            x = self.neck(x)
        x = self.head(x)
        if self.training:
            x = self.loss(x, labels)
        else:
            x = self.decoder(x)
        return x
