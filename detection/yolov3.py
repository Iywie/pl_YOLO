from pytorch_lightning import LightningModule

from data import TrainTransform, ValTransform
import models.backbones as BACKBONE
import models.necks as NECK
import models.heads as HEAD
from models.evaluators.coco import COCOEvaluator, convert_to_coco_format
from models.evaluators.post_process import coco_post

from torch.optim import Adam, SGD
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
        # anchors_mask = self.decoder_cfgs['ANCHORS_MASK']
        self.strides = [8, 16, 32]
        # loss parameters
        self.use_l1 = False
        # evaluate parameters
        self.nms_threshold = 0.65
        self.confidence_threshold = 0.01
        # dataloader parameters
        self.img_size_train = tuple(self.dataset_cfgs['TRAIN_SIZE'])
        self.img_size_val = tuple(self.dataset_cfgs['VAL_SIZE'])
        self.train_batch_size = self.dataset_cfgs['TRAIN_BATCH_SIZE']
        self.val_batch_size = self.dataset_cfgs['VAL_BATCH_SIZE']
        self.val_dataset = None
        # --------------- transform config ----------------- #
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
        self.head = HEAD.DecoupledHead(self.num_classes, n_anchors, n_channels, n_norm, n_act)
        # self.decoder = HEAD.YOLOv3Decoder(self.num_classes, n_anchors, self.anchors, self.strides)
        self.loss = []
        for i in range(3):
            self.loss.append(HEAD.YOLOv3Loss(self.anchors[i], self.num_classes, self.img_size_train))
        self.head.initialize_biases(1e-2)

    def forward(self, x):
        return x

    def training_step(self, batch, batch_idx):
        imgs, labels, _, _, _ = batch
        output = self.backbone(imgs)
        output = self.neck(output)
        output = self.head(output)
        loss = 0
        for i in range(len(output)):
            _loss = self.loss[i](output[i], labels)
            loss += _loss
        # losses_name = ["total_loss", "x", "y", "w", "h", "conf", "cls"]
        # losses = []
        # for _ in range(len(losses_name)):
        #     losses.append([])
        # for i in range(3):
        #     _loss_item = self.loss[i](output[i], labels)
        #     for j, l in enumerate(_loss_item):
        #         losses[j].append(l)
        # losses = [sum(l) for l in losses]
        # loss = losses[0]

        # pred, maps_h, maps_w = self.decoder(output)
        # loss, iou_loss, conf_loss, cls_loss, l1_loss, num_fg = HEAD.YOLOv3Loss(
        #     labels, pred, self.num_classes, self.anchors, self.strides, maps_h, maps_w)
        self.lr_scheduler.step()
        self.log("lr", self.lr_scheduler.optimizer.param_groups[0]['lr'], prog_bar=True)
        # self.log("metrics/batch/iou_loss", iou_loss, prog_bar=False)
        # self.log("metrics/batch/l1_loss", l1_loss, prog_bar=False)
        # self.log("metrics/batch/conf_loss", conf_loss, prog_bar=False)
        # self.log("metrics/batch/cls_loss", cls_loss, prog_bar=False)
        # self.log("metrics/batch/num_fg", num_fg, prog_bar=False)
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
        detections = convert_to_coco_format(detections, image_id, img_hw, self.img_size_val, self.val_dataset.class_ids)
        return detections

    def validation_epoch_end(self, validation_step_outputs):
        detect_list = []
        for i in range(len(validation_step_outputs)):
            detect_list += validation_step_outputs[i]
        ap50_95, ap50, summary = COCOEvaluator(
            detect_list, self.val_dataset)
        print("Batch {:d}, mAP = {:.3f}, mAP50 = {:.3f}".format(self.current_epoch, ap50_95, ap50))
        print(summary)
        self.log("metrics/evaluate/mAP", ap50_95, prog_bar=False)
        self.log("metrics/evaluate/mAP50", ap50, prog_bar=False)

    def configure_optimizers(self):
        optimizer = SGD(self.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
        steps_per_epoch = 1440 // self.train_batch_size
        self.lr_scheduler = CosineWarmupScheduler(
            optimizer, warmup=self.warmup * steps_per_epoch, max_iters=self.trainer.max_epochs * steps_per_epoch
        )
        return optimizer

    def train_dataloader(self):
        from data.datasets.cocoDataset import COCODataset
        from torch.utils.data.dataloader import DataLoader
        data_dir = self.dataset_cfgs['DIR']
        json = self.dataset_cfgs['TRAIN_JSON']
        dataset = COCODataset(
            data_dir,
            json,
            name='train',
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
        from data.datasets.cocoDataset import COCODataset
        from torch.utils.data.dataloader import DataLoader
        data_dir = self.dataset_cfgs['DIR']
        json = self.dataset_cfgs['VAL_JSON']
        self.val_dataset = COCODataset(
            data_dir,
            json,
            name="val",
            img_size=self.img_size_val,
            preprocess=ValTransform(legacy=False),
        )
        val_loader = DataLoader(self.val_dataset, batch_size=self.val_batch_size, num_workers=4, shuffle=False)
        return val_loader
