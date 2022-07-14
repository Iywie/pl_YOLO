import torch
import time
import torch.nn as nn
from pytorch_lightning import LightningModule

# Model
from models.detectors.OneStage import OneStageD
from models.backbones.darknet_csp import CSPDarkNet
from models.necks.pafpn_csp import CSPPAFPN
from models.heads.decoupled_head import DecoupledHead
from models.heads.yolov5.yolov5_loss import YOLOv5Loss
from models.heads.yolov5.yolov5_decoder import YOLOv5Decoder
from models.evaluators.coco import COCOEvaluator, convert_to_coco_format

from torch.optim import SGD, AdamW, Adam


class LitYOLOv5(LightningModule):
    def __init__(self, cfgs):
        super().__init__()
        self.cb = cfgs['backbone']
        self.cn = cfgs['neck']
        self.ch = cfgs['head']
        self.cd = cfgs['dataset']
        self.co = cfgs['optimizer']
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
        self.anchors = self.ch['anchors']
        n_anchors = len(self.anchors)
        self.strides = [8, 16, 32]
        anchor_thre = 4.0
        balance = [4.0, 1.0, 0.4]
        # evaluate parameters
        self.nms_threshold = 0.65
        self.confidence_threshold = 0.01
        # data
        self.num_classes = self.cd['num_classes']
        self.train_batch_size = self.cd['train_batch_size']
        self.val_batch_size = self.cd['val_batch_size']
        self.img_size_train = self.cd['train_size']
        self.img_size_val = tuple(self.cd['val_size'])
        # Training
        self.warmup = self.co['warmup']
        self.iter_times = []
        # Model
        self.backbone = CSPDarkNet(b_depth, b_channels, out_features, b_norm, b_act)
        self.neck = CSPPAFPN(n_depth, n_channels, n_norm, n_act)
        self.head = DecoupledHead(self.num_classes, n_anchors, n_channels, n_norm, n_act)
        self.loss = YOLOv5Loss(self.num_classes, self.img_size_train, self.anchors, self.strides,
                               anchor_thre, balance)
        self.decoder = YOLOv5Decoder(self.num_classes, self.anchors, self.strides)
        self.model = OneStageD(self.backbone, self.neck, self.head)
        self.automatic_optimization = False

    def training_step(self, batch, batch_idx):
        imgs, labels, _, _, _ = batch
        output = self.model(imgs)
        loss, _ = self.loss(output, labels)
        self.log("loss/loss", loss, prog_bar=True)
        self.log("lr", self.trainer.optimizers[0].param_groups[0]['lr'], prog_bar=True)
        # Backward
        opt = self.optimizers()
        opt.zero_grad()
        self.manual_backward(loss)
        opt.step()
        self.lr_schedulers().step()

    def validation_step(self, batch, batch_idx):
        imgs, labels, img_hw, image_id, img_name = batch
        model = self.model
        start_time = time.time()
        output = model(imgs)
        detections = self.decoder(output, self.confidence_threshold, self.nms_threshold)
        self.iter_times.append(time.time() - start_time)
        detections = convert_to_coco_format(detections, image_id, img_hw, self.img_size_val,
                                            self.trainer.datamodule.dataset_val.class_ids)
        return detections

    def validation_epoch_end(self, val_step_outputs):
        detect_list = []
        for i in range(len(val_step_outputs)):
            detect_list += val_step_outputs[i]
        ap50_95, ap50, summary = COCOEvaluator(
            detect_list, self.trainer.datamodule.dataset_val)
        print("Batch {:d}, mAP = {:.3f}, mAP50 = {:.3f}".format(self.current_epoch, ap50_95, ap50))
        print(summary)
        self.log("val/mAP", ap50_95, prog_bar=False)
        self.log("val/mAP50", ap50, prog_bar=False)

    def configure_optimizers(self):
        optimizer = SGD(self.parameters(), lr=self.co["learning_rate"], momentum=self.co["momentum"])
        lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=self.co["learning_rate"],
                                                           pct_start=self.warmup,
                                                           total_steps=self.trainer.estimated_stepping_batches)
        return [optimizer], [lr_scheduler]

    def on_train_end(self) -> None:
        average_ifer_time = torch.tensor(self.iter_times, dtype=torch.float32).mean()
        print("The average iference time is ", average_ifer_time, " ms")

