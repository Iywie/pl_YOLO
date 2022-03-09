import torch
import torch.nn as nn
from pytorch_lightning import LightningModule

from models.data.datasets.cocoDataset import COCODataset
from torch.utils.data.dataloader import DataLoader
from models.data.augmentation.data_augments import TrainTransform, ValTransform
# Model
from models.detectors.OneStage import OneStageD
from models.backbones.darknet_csp import CSPDarkNet
from models.necks.pafpn import PAFPN
from models.heads.decoupled_head import DecoupledHead
from models.heads.yolov5.yolov5_loss import YOLOv5Loss
from models.heads.yolov5.yolov5_decoder import YOLOv5Decoder
from models.evaluators.coco import COCOEvaluator, convert_to_coco_format
from models.data.mosaic_detection import MosaicDetection
from torch.utils.data.sampler import BatchSampler, RandomSampler
from models.lr_scheduler import CosineWarmupScheduler
from models.utils.ema import ModelEMA


class LitYOLOv5(LightningModule):
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
        self.anchors = self.ch['anchors']
        n_anchors = len(self.anchors)
        self.strides = [8, 16, 32]
        anchor_thre = 4.0
        balance = [4.0, 1.0, 0.4]
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
        # Training parameters
        self.warmup = 5
        # --------------- transform config ----------------- #
        self.mosaic_epoch = self.ct['mosaic_epoch']
        self.mosaic_prob = 1.0
        self.mixup_prob = 1.0
        self.hsv_prob = 1.0
        self.flip_prob = 0.5
        self.degrees = 10.0
        self.translate = 0.1
        self.mosaic_scale = (0.1, 2)
        self.mixup_scale = (0.5, 1.5)
        self.shear = 2.0
        self.perspective = 0.0
        self.enable_mixup = True
        # Model
        self.backbone = CSPDarkNet(b_depth, b_channels, out_features, b_norm, b_act)
        self.neck = PAFPN(n_depth, n_channels, n_norm, n_act)
        self.head = DecoupledHead(self.num_classes, n_anchors, n_channels, n_norm, n_act)
        self.loss = YOLOv5Loss(self.num_classes, self.img_size_train, self.anchors, self.strides,
                               anchor_thre, balance)
        self.decoder = YOLOv5Decoder(self.num_classes, self.anchors, self.strides)

        self.model = OneStageD(self.backbone, self.neck, self.head)
        self.ema = self.co['ema']
        self.ema_model = None

    def on_train_start(self):
        if self.ema:
            self.ema_model = ModelEMA(self.model, 0.9998)
            self.ema_model.updates = len(self.dataset_train) * self.current_epoch

    def training_step(self, batch, batch_idx):
        imgs, labels, _, _, _ = batch
        output = self.model(imgs)
        loss, _ = self.loss(output, labels)
        self.log("loss/loss", loss, prog_bar=False)
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

    def validation_step(self, batch, batch_idx):
        imgs, _, img_hw, image_id, img_name = batch
        output = self.model(imgs)
        detections = self.decoder(output, self.confidence_threshold, self.nms_threshold, multi_label=False)
        detections = convert_to_coco_format(detections, image_id, img_hw,
                                            self.img_size_val, self.dataset_val.class_ids)
        return detections

    def validation_epoch_end(self, results):
        detect_list = []
        for i in range(len(results)):
            detect_list += results[i]
        ap50_95, ap50, summary = COCOEvaluator(
            detect_list, self.dataset_val)
        print(summary)
        self.log("val/mAP", ap50_95, prog_bar=False)
        self.log("val/mAP50", ap50, prog_bar=False)

    def configure_optimizers(self):
        pg0, pg1, pg2 = [], [], []  # optimizer parameter groups
        for k, v in self.model.named_modules():
            if hasattr(v, "bias") and isinstance(v.bias, nn.Parameter):
                pg2.append(v.bias)  # biases
            if isinstance(v, nn.BatchNorm2d) or "bn" in k:
                pg0.append(v.weight)  # no decay
            elif hasattr(v, "weight") and isinstance(v.weight, nn.Parameter):
                pg1.append(v.weight)  # apply decay
        optimizer = torch.optim.SGD(
            pg0, lr=self.co["learning_rate"], momentum=self.co["momentum"], nesterov=True
        )
        optimizer.add_param_group(
            {"params": pg1, "weight_decay": self.co["weight_decay"]}
        )  # add pg1 with weight_decay
        optimizer.add_param_group({"params": pg2})
        # optimizer = SGD(self.parameters(), lr=0.01, momentum=0.9, weight_decay=4e-05)
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
            cache=True
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
            preprocess=ValTransform(legacy=False, max_labels=50),
            cache=False,
        )
        sampler = torch.utils.data.SequentialSampler(self.dataset_val)
        val_loader = DataLoader(self.dataset_val, batch_size=self.val_batch_size, sampler=sampler,
                                num_workers=6, pin_memory=True, shuffle=False)
        return val_loader

