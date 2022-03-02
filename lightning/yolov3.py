from pytorch_lightning import LightningModule

import models.backbones as BACKBONE
import models.necks as NECK
import models.heads as HEAD
from models.evaluators.coco import COCOEvaluator, convert_to_coco_format
from models.evaluators.post_process import coco_post

from models.data.datasets.cocoDataset import COCODataset
from torch.utils.data.dataloader import DataLoader
from models.data.data_augments import TrainTransform, ValTransform
from torch.optim import SGD
from models.lr_scheduler import CosineWarmupScheduler


class LitYOLOv3(LightningModule):

    def __init__(self, cfgs):
        super().__init__()
        self.backbone_cfgs = cfgs['CSPDARKNET']
        self.neck_cfgs = cfgs['PAFPN']
        self.head_cfgs = cfgs['DECOUPLEDHEAD']
        self.decoder_cfgs = cfgs['DECODER']
        self.dataset_cfgs = cfgs['DATASET']
        # backbone parameters
        b_depth = self.backbone_cfgs['DEPTH']
        b_norm = self.backbone_cfgs['NORM']
        b_act = self.backbone_cfgs['ACT']
        b_channels = self.backbone_cfgs['INPUT_CHANNELS']
        out_features = self.backbone_cfgs['OUT_FEATURES']
        # neck parameters
        n_depth = self.neck_cfgs['DEPTH']
        n_channels = self.neck_cfgs['INPUT_CHANNELS']
        n_norm = self.neck_cfgs['NORM']
        n_act = self.neck_cfgs['ACT']
        # head parameters
        self.num_classes = self.head_cfgs['CLASSES']
        # decoder parameters
        self.anchors = self.decoder_cfgs['ANCHORS']
        n_anchors = len(self.anchors)
        self.strides = [8, 16, 32]
        # loss parameters
        self.use_l1 = False
        # evaluate parameters
        self.nms_threshold = 0.7
        self.confidence_threshold = 0.2
        # dataloader parameters
        self.data_dir = self.dataset_cfgs['DIR']
        self.train_dir = self.dataset_cfgs['TRAIN']
        self.val_dir = self.dataset_cfgs['VAL']
        self.img_size_train = tuple(self.dataset_cfgs['TRAIN_SIZE'])
        self.img_size_val = tuple(self.dataset_cfgs['VAL_SIZE'])
        self.train_batch_size = self.dataset_cfgs['TRAIN_BATCH_SIZE']
        self.val_batch_size = self.dataset_cfgs['VAL_BATCH_SIZE']
        self.val_dataset = None
        # Training
        self.warmup = 3

        self.backbone = BACKBONE.CSPDarkNet(b_depth, b_channels, out_features, b_norm, b_act)
        self.neck = NECK.PAFPN(n_depth, out_features, n_channels, n_norm, n_act)
        self.head = HEAD.DecoupledHead(self.num_classes, n_anchors, n_channels, n_norm, n_act)
        self.loss = []
        for i in range(3):
            self.loss.append(HEAD.YOLOv3Loss(self.anchors[i], self.num_classes, self.img_size_train))

    def training_step(self, batch, batch_idx):
        imgs, labels, _, _, _ = batch
        output = self.backbone(imgs)
        output = self.neck(output)
        output = self.head(output)
        loss = 0
        for i in range(len(output)):
            _loss = self.loss[i](output[i], labels)
            loss += _loss

        self.lr_scheduler.step()
        self.log("lr", self.lr_scheduler.optimizer.param_groups[0]['lr'], prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        imgs, labels, img_hw, image_id, img_name = batch
        output = self.backbone(imgs)
        output = self.neck(output)
        output = self.head(output)
        pred = []
        for i in range(len(output)):
            _loss = self.loss[i](output[i])
            pred.append(_loss)
        detections = coco_post(pred, self.num_classes, self.confidence_threshold, self.nms_threshold)
        # map50_batch, correct, num_gt_batch = MyEvaluator_step(detections, labels, img_hw, image_id, self.img_size_val)
        # print('AP50 of batch %d: %.5f' % (batch_idx, map50_batch))
        detections = convert_to_coco_format(detections, image_id, img_hw,
                                            self.img_size_val, self.val_dataset.class_ids)
        # return correct, num_gt_batch, detections
        return detections

    def validation_epoch_end(self, results):
        corrects = 0
        num_gts = 0
        detect_list = []
        # for i in range(len(results)):
        #     corrects += results[i][0]
        #     num_gts += results[i][1]
        #     detect_list += results[i][2]
        # print("Epoch:%d: Average Precision = %.5f" % (self.current_epoch, float(corrects / num_gts)))
        for i in range(len(results)):
            detect_list += results[i]
        ap50_95, ap50, summary = COCOEvaluator(
            detect_list, self.val_dataset)
        print(summary)

    def configure_optimizers(self):
        optimizer = SGD(self.parameters(), lr=0.001, momentum=0.9, weight_decay=4e-05)
        steps_per_epoch = 1440 // self.train_batch_size
        self.lr_scheduler = CosineWarmupScheduler(
            optimizer, warmup=self.warmup * steps_per_epoch, max_iters=self.trainer.max_epochs * steps_per_epoch
        )
        return optimizer

    def train_dataloader(self):
        dataset = COCODataset(
            self.data_dir,
            name=self.train_dir,
            img_size=self.img_size_train,
            preprocess=TrainTransform(
                max_labels=50,
                flip_prob=0,
                hsv_prob=0
            )
        )
        train_loader = DataLoader(dataset, batch_size=self.train_batch_size, num_workers=4, shuffle=False)
        return train_loader

    def val_dataloader(self):
        self.val_dataset = COCODataset(
            self.data_dir,
            name=self.val_dir,
            img_size=self.img_size_val,
            preprocess=ValTransform(legacy=False, max_labels=50)
        )
        val_loader = DataLoader(self.val_dataset, batch_size=self.val_batch_size, num_workers=4, shuffle=False)
        return val_loader
