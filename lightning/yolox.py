import torch
import torch.nn as nn
from pytorch_lightning import LightningModule
# Model
from models.detectors.OneStage import OneStageD
from models.backbones.darknet_csp import CSPDarkNet
from models.necks.pafpn import PAFPN
from models.heads.decoupled_head import DecoupledHead
from models.heads.yolox.yolox_loss import YOLOXLoss
from models.heads.yolox.yolox_decoder import YOLOXDecoder
from models.heads.yolox.yolox_head import YOLOXHead

from models.evaluators.coco import COCOEvaluator, convert_to_coco_format
# Data
from models.data.datasets.cocoDataset import COCODataset
from models.data.samplers import InfiniteSampler, YoloBatchSampler
from models.data.mosaic_detection import MosaicDetection
from torch.utils.data.dataloader import DataLoader
from models.data.data_augments import TrainTransform, ValTransform
from torch.utils.data.sampler import BatchSampler, SequentialSampler, RandomSampler
from torch.optim import SGD
from models.lr_scheduler import CosineWarmupScheduler
from models.utils.ema import ModelEMA
import torchvision.transforms as transforms


class LitYOLOX(LightningModule):

    def __init__(self, cfgs):
        super().__init__()
        self.cb = cfgs['backbone']
        self.cn = cfgs['neck']
        self.ch = cfgs['head']
        self.cd = cfgs['dataset']
        self.co = cfgs['optimizer']
        self.ct = cfgs['transform']
        # backbone parameters
        b_depth = self.cb['depth']
        b_norm = self.cb['normalization']
        b_act = self.cb['activation']
        b_channels = self.cb['input_channels']
        out_features = self.cb['output_features']
        # neck parameters
        n_depth = self.cn['depth']
        n_channels = self.cn['input_channels']
        n_norm = self.cn['normalization']
        n_act = self.cn['activation']
        # head parameters
        self.num_classes = self.ch['classes']
        n_anchors = 1
        strides = [8, 16, 32]
        # loss parameters
        self.use_l1 = False
        # evaluate parameters
        self.nms_threshold = 0.65
        self.confidence_threshold = 0.01
        # dataloader parameters
        self.data_dir = self.cd['dir']
        self.train_dir = self.cd['train']
        self.val_dir = self.cd['val']
        self.img_size_train = tuple(self.cd['train_size'])
        self.img_size_val = tuple(self.cd['val_size'])
        self.train_batch_size = self.cd['train_batch_size']
        self.val_batch_size = self.cd['val_batch_size']
        self.dataset_val = None
        self.dataset_train = None
        # --------------- transform config ----------------- #
        self.mosaic_epoch = self.ct['mosaic_epoch']
        self.mosaic_prob = self.ct['mosaic_prob']
        self.mixup_prob = self.ct['mixup_prob']
        self.hsv_prob = self.ct['hsv_prob']
        self.flip_prob = self.ct['flip_prob']
        self.degrees = self.ct['degrees']
        self.translate = self.ct['translate']
        self.mosaic_scale = self.ct['mosaic_scale']
        self.mixup_scale = self.ct['mixup_scale']
        self.shear = self.ct['shear']
        self.perspective = self.ct['perspective']
        self.enable_mixup = self.ct['enable_mixup']
        # Training
        self.warmup = self.co['warmup']

        self.backbone = CSPDarkNet(b_depth, b_channels, out_features, b_norm, b_act)
        self.neck = PAFPN(n_depth, n_channels, n_norm, n_act)
        self.head = DecoupledHead(self.num_classes, n_anchors, n_channels, n_norm, n_act)
        self.loss = YOLOXLoss(self.num_classes, strides)
        self.decoder = YOLOXDecoder(self.num_classes, strides)
        self.test_head = YOLOXHead(self.num_classes, strides, n_channels, n_act)

        self.model = OneStageD(self.backbone, self.neck, self.head)
        self.ema = self.co['ema']
        self.ema_model = None

        self.head.initialize_biases(1e-2)
        self.model.apply(initializer)

    def on_train_start(self):
        if self.ema:
            self.ema_model = ModelEMA(self.model, 0.9998)
            self.ema_model.updates = len(self.dataset_train) * self.current_epoch

    def training_step(self, batch, batch_idx):
        imgs, labels, _, _, _ = batch
        # _, _, h, w = imgs.shape
        # # perform augmentations with YOCO
        # images = YOCO(imgs, aug, h, w)
        output = self.model(imgs)
        loss, loss_iou, loss_obj, loss_cls, loss_l1, proportion = self.loss(output, labels)
        self.log("loss/loss", loss, prog_bar=False)
        self.log("loss/iou", loss_iou, prog_bar=False)
        self.log("loss/obj", loss_obj, prog_bar=False)
        self.log("loss/cls", loss_cls, prog_bar=False)
        self.log("loss/l1", loss_l1, prog_bar=False)
        self.log("loss/proportion", proportion, prog_bar=False)
        return loss

    def on_train_batch_end(self, outputs, batch, batch_idx, unused=0):
        self.lr_scheduler.step()
        self.log("lr", self.lr_scheduler.optimizer.param_groups[0]['lr'], prog_bar=True)
        if self.ema:
            self.ema_model.update(self.model)

    def training_epoch_end(self, outputs):
        if self.current_epoch < self.mosaic_epoch:
            self.dataset_train.enable_mosaic = True
        else:
            self.dataset_train.enable_mosaic = False
            self.loss.use_l1 = True

    def validation_step(self, batch, batch_idx):
        imgs, labels, img_hw, image_id, img_name = batch
        if self.ema_model is not None:
            model = self.ema_model.ema
        else:
            model = self.model
        output = model(imgs)
        detections = self.decoder(output, self.confidence_threshold, self.nms_threshold)
        detections = convert_to_coco_format(detections, image_id, img_hw, self.img_size_val, self.dataset_val.class_ids)
        return detections

    def validation_epoch_end(self, validation_step_outputs):
        detect_list = []
        for i in range(len(validation_step_outputs)):
            detect_list += validation_step_outputs[i]
        ap50_95, ap50, summary = COCOEvaluator(
            detect_list, self.dataset_val)
        print("Batch {:d}, mAP = {:.3f}, mAP50 = {:.3f}".format(self.current_epoch, ap50_95, ap50))
        print(summary)
        self.log("val/mAP", ap50_95, prog_bar=False)
        self.log("val/mAP50", ap50, prog_bar=False)

    def configure_optimizers(self):
        # pg0, pg1, pg2 = [], [], []  # optimizer parameter groups
        # for k, v in self.model.named_modules():
        #     if hasattr(v, "bias") and isinstance(v.bias, nn.Parameter):
        #         pg2.append(v.bias)  # biases
        #     if isinstance(v, nn.BatchNorm2d) or "bn" in k:
        #         pg0.append(v.weight)  # no decay
        #     elif hasattr(v, "weight") and isinstance(v.weight, nn.Parameter):
        #         pg1.append(v.weight)  # apply decay
        # optimizer = torch.optim.SGD(
        #     pg0, lr=self.co["learning_rate"], momentum=self.co["momentum"], nesterov=True
        # )
        # optimizer.add_param_group(
        #     {"params": pg1, "weight_decay": self.co["weight_decay"]}
        # )  # add pg1 with weight_decay
        # optimizer.add_param_group({"params": pg2})

        optimizer = SGD(self.parameters(), lr=self.co["learning_rate"], momentum=self.co["momentum"])
        steps_per_epoch = 1440 // self.train_batch_size
        self.lr_scheduler = CosineWarmupScheduler(
            optimizer, warmup=self.warmup * steps_per_epoch, max_iters=self.trainer.max_epochs * steps_per_epoch
        )
        return optimizer

    def train_dataloader(self):
        self.dataset_train = COCODataset(
            self.data_dir,
            name=self.train_dir,
            img_size=self.img_size_train,
            preprocess=TrainTransform(max_labels=50, flip_prob=self.flip_prob, hsv_prob=self.hsv_prob),
            cache=False
        )
        self.dataset_train = MosaicDetection(
            self.dataset_train,
            mosaic=False,
            img_size=self.img_size_train,
            preprocess=TrainTransform(
                max_labels=120,
                flip_prob=self.flip_prob,
                hsv_prob=self.hsv_prob),
            degrees=self.degrees,
            translate=self.translate,
            mosaic_scale=self.mosaic_scale,
            mixup_scale=self.mixup_scale,
            shear=self.shear,
            perspective=self.perspective,
            enable_mixup=self.enable_mixup,
            mosaic_prob=self.mosaic_prob,
            mixup_prob=self.mixup_prob,
        )
        sampler = RandomSampler(self.dataset_train)
        batch_sampler = BatchSampler(sampler, batch_size=self.train_batch_size, drop_last=False)
        train_loader = DataLoader(self.dataset_train, batch_sampler=batch_sampler,
                                  num_workers=12, pin_memory=True)
        return train_loader

    def val_dataloader(self):
        self.dataset_val = COCODataset(
            self.data_dir,
            name=self.val_dir,
            img_size=self.img_size_val,
            preprocess=ValTransform(legacy=False),
            cache=False,
        )
        sampler = torch.utils.data.SequentialSampler(self.dataset_val)
        val_loader = DataLoader(self.dataset_val, batch_size=self.val_batch_size, sampler=sampler,
                                num_workers=6, pin_memory=True, shuffle=False)
        return val_loader


def initializer(M):
    for m in M.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.eps = 1e-3
            m.momentum = 0.03


aug = torch.nn.Sequential(
            transforms.RandomHorizontalFlip(), )


def YOCO(images, aug, h, w):
    images = torch.cat((aug(images[:, :, :, 0:int(w / 2)]), aug(images[:, :, :, int(w / 2):w])), dim=3) if \
        torch.rand(1) > 0.5 else torch.cat((aug(images[:, :, 0:int(h / 2), :]), aug(images[:, :, int(h / 2):h, :])),
                                           dim=2)
    return images
