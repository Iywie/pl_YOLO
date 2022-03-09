from pytorch_lightning import LightningModule

from models.backbones.darknet_csp import CSPDarkNet
from models.necks.pafpn import PAFPN
from models.heads.decoupled_head import DecoupledHead
from models.heads.yolov3.yolov3_loss import YOLOv3Loss
from models.detectors.OneStage import OneStageD
from models.evaluators.coco import COCOEvaluator, convert_to_coco_format
from models.evaluators.post_process import coco_post

from models.data.datasets.cocoDataset import COCODataset
from torch.utils.data.dataloader import DataLoader
from models.data.augmentation.data_augments import TrainTransform, ValTransform
from torch.optim import SGD
from models.lr_scheduler import CosineWarmupScheduler

from models.utils.ema import ModelEMA


class LitYOLOv3(LightningModule):

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
        # evaluate parameters
        self.nms_threshold = 0.7
        self.confidence_threshold = 0.2
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
        # Training
        self.warmup = self.co['warmup']

        self.backbone = CSPDarkNet(b_depth, b_channels, out_features, b_norm, b_act)
        self.neck = PAFPN(n_depth, out_features, n_channels, n_norm, n_act)
        self.head = DecoupledHead(self.num_classes, n_anchors, n_channels, n_norm, n_act)
        self.loss = []
        for i in range(3):
            self.loss.append(YOLOv3Loss(self.anchors[i], self.num_classes, self.img_size_train))

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
        loss = 0
        for i in range(len(output)):
            _loss = self.loss[i](output[i], labels)
            loss += _loss
        return loss

    def on_train_batch_end(self, outputs, batch, batch_idx, unused=0):
        self.lr_scheduler.step()
        self.log("lr", self.lr_scheduler.optimizer.param_groups[0]['lr'], prog_bar=True)
        if self.ema:
            self.ema_model.update(self.model)

    def training_epoch_end(self, outputs):
        if self.current_epoch > self.mosaic_epoch:
            self.dataset_train.enable_mosaic = False

    def validation_step(self, batch, batch_idx):
        imgs, labels, img_hw, image_id, img_name = batch
        if self.ema:
            model = self.ema_model.ema
        else:
            model = self.model
        output = model(imgs)
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
