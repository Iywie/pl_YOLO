from pytorch_lightning import LightningModule
import torch
import torch.nn as nn

import models.backbones as BACKBONE
import models.necks as NECK
from models.heads.decoupled_head import DecoupledHead
from models.heads.yolox.yolox_loss import YOLOXLoss
from models.heads.yolox.yolox_decoder import YOLOXDecoder
from models.heads.yolox.yolox_head import YOLOXHead
from models.evaluators.coco_evaluator_mine import MyEvaluator_step
from models.evaluators.coco import COCOEvaluator, convert_to_coco_format
from models.evaluators.post_process import coco_post

from models.data.samplers import InfiniteSampler, YoloBatchSampler
from models.data.datasets.cocoDataset import COCODataset
from models.data.mosaic_detection import MosaicDetection
from torch.utils.data.dataloader import DataLoader
from models.data.data_augments import TrainTransform, ValTransform
from torch.optim import SGD
from models.lr_scheduler import CosineWarmupScheduler
from torch.utils.data.sampler import BatchSampler, SequentialSampler


class LitYOLOX(LightningModule):

    def __init__(self, cfgs):
        super().__init__()
        self.cfg_backbone = cfgs['backbone']
        self.cfg_neck = cfgs['neck']
        self.cfg_head = cfgs['head']
        self.cfg_dataset = cfgs['dataset']
        self.cfg_transform = cfgs['transform']
        # backbone parameters
        b_depth = self.cfg_backbone['depth']
        b_norm = self.cfg_backbone['normalization']
        b_act = self.cfg_backbone['activation']
        b_channels = self.cfg_backbone['input_channels']
        out_features = self.cfg_backbone['output_features']
        # neck parameters
        n_depth = self.cfg_neck['depth']
        n_channels = self.cfg_neck['input_channels']
        n_norm = self.cfg_neck['normalization']
        n_act = self.cfg_neck['activation']
        # head parameters
        self.num_classes = self.cfg_head['classes']
        n_anchors = 1
        strides = [8, 16, 32]
        # loss parameters
        self.use_l1 = False
        # evaluate parameters
        self.nms_threshold = 0.65
        self.confidence_threshold = 0.01
        # dataloader parameters
        self.data_dir = self.cfg_dataset['dir']
        self.train_dir = self.cfg_dataset['train']
        self.val_dir = self.cfg_dataset['val']
        self.img_size_train = tuple(self.cfg_dataset['train_size'])
        self.img_size_val = tuple(self.cfg_dataset['val_size'])
        self.train_batch_size = self.cfg_dataset['train_batch_size']
        self.val_batch_size = self.cfg_dataset['val_batch_size']
        self.dataset_val = None
        self.dataset_train = None
        # --------------- transform config ----------------- #
        self.mosaic_epoch = self.cfg_transform['mosaic_epoch']
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
        # Training
        self.warmup = 5

        self.backbone = BACKBONE.CSPDarkNet(b_depth, b_channels, out_features, b_norm, b_act)
        self.neck = NECK.PAFPN(n_depth, out_features, n_channels, n_norm, n_act)
        self.head = DecoupledHead(self.num_classes, n_anchors, n_channels, n_norm, n_act)
        self.head.initialize_biases(1e-2)
        self.loss = YOLOXLoss(self.num_classes, strides)
        self.decoder = YOLOXDecoder(self.num_classes, strides)
        self.test_head = YOLOXHead(self.num_classes, strides, n_channels, n_act)

        self.models = [self.backbone, self.neck, self.head]

        for model in self.models:
            model.apply(initializer)

    def training_step(self, batch, batch_idx):
        imgs, labels, _, _, _ = batch
        output = self.backbone(imgs)
        output = self.neck(output)
        output = self.head(output)
        outputs = self.loss(output, labels)
        loss = outputs[0]
        self.lr_scheduler.step()
        self.log("lr", self.lr_scheduler.optimizer.param_groups[0]['lr'], prog_bar=True)
        return loss

    def training_epoch_end(self, outputs):
        pass

    def validation_step(self, batch, batch_idx):
        imgs, labels, img_hw, image_id, img_name = batch
        output = self.backbone(imgs)
        output = self.neck(output)
        outputs = self.head(output)
        detections = self.decoder(outputs, self.confidence_threshold, self.nms_threshold)
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
        # self.log("metrics/evaluate/mAP", ap50_95, prog_bar=False)
        # self.log("metrics/evaluate/mAP50", ap50, prog_bar=False)

    def configure_optimizers(self):
        # pg0, pg1, pg2 = [], [], []  # optimizer parameter groups
        # for model in self.models:
        #     for k, v in model.named_modules():
        #         if hasattr(v, "bias") and isinstance(v.bias, nn.Parameter):
        #             pg2.append(v.bias)  # biases
        #         if isinstance(v, nn.BatchNorm2d) or "bn" in k:
        #             pg0.append(v.weight)  # no decay
        #         elif hasattr(v, "weight") and isinstance(v.weight, nn.Parameter):
        #             pg1.append(v.weight)  # apply decay
        # optimizer = torch.optim.SGD(
        #     pg0, lr=0.01, momentum=0.9, nesterov=True
        # )
        # optimizer.add_param_group(
        #     {"params": pg1, "weight_decay": 5e-4}
        # )  # add pg1 with weight_decay
        # optimizer.add_param_group({"params": pg2})

        optimizer = SGD(self.parameters(), lr=0.03, momentum=0.9)
        steps_per_epoch = 1440 // self.train_batch_size
        self.lr_scheduler = CosineWarmupScheduler(
            optimizer, warmup=self.warmup * steps_per_epoch, max_iters=self.trainer.max_epochs * steps_per_epoch
        )
        return optimizer

    def train_dataloader(self):
        dataset_train = COCODataset(
            self.data_dir,
            name=self.train_dir,
            img_size=self.img_size_train,
            preprocess=TrainTransform(max_labels=50, flip_prob=0, hsv_prob=0),
            cache=False
        )
        # dataset_train = MosaicDetection(
        #     dataset_train,
        #     mosaic=True,
        #     img_size=self.img_size_train,
        #     preprocess=TrainTransform(
        #         max_labels=120,
        #         flip_prob=self.flip_prob,
        #         hsv_prob=self.hsv_prob),
        #     degrees=self.degrees,
        #     translate=self.translate,
        #     mosaic_scale=self.mosaic_scale,
        #     mixup_scale=self.mixup_scale,
        #     shear=self.shear,
        #     perspective=self.perspective,
        #     enable_mixup=self.enable_mixup,
        #     mosaic_prob=self.mosaic_prob,
        #     mixup_prob=self.mixup_prob,
        # )

        sampler = InfiniteSampler(len(dataset_train), seed=0)
        # sampler = SequentialSampler(dataset_train)

        # batch_sampler = YoloBatchSampler(
        #     sampler=sampler,
        #     batch_size=self.train_batch_size,
        #     drop_last=False,
        #     mosaic=True,
        # )
        batch_sampler = BatchSampler(sampler, batch_size=self.train_batch_size, drop_last=False,)
        train_loader = DataLoader(dataset_train, batch_sampler=batch_sampler,
                                  num_workers=6, pin_memory=True, shuffle=False)

        # sampler = torch.utils.data.SequentialSampler(self.dataset_train)
        # train_loader = DataLoader(self.dataset_train, batch_size=self.train_batch_size, sampler=sampler,
        #                           num_workers=4, pin_memory=True, shuffle=False)

        return train_loader

    def val_dataloader(self):
        self.dataset_val = COCODataset(
            self.data_dir,
            name=self.val_dir,
            img_size=self.img_size_val,
            preprocess=ValTransform(legacy=False),
        )
        sampler = torch.utils.data.SequentialSampler(self.dataset_val)
        val_loader = DataLoader(self.dataset_val, batch_size=self.val_batch_size, sampler=sampler,
                                num_workers=4, pin_memory=True, shuffle=False)
        return val_loader


def initializer(M):
    for m in M.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.eps = 1e-3
            m.momentum = 0.03
