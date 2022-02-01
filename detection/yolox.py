from pytorch_lightning import LightningModule

from data import TrainTransform, ValTransform
import models.backbones as BACKBONE
import models.necks as NECK
import models.heads as HEAD
from models.evaluators.coco import COCOEvaluator, postprocess

from torch.optim import Adam


class LitYOLOX(LightningModule):

    def __init__(self, cfgs):
        super().__init__()

        self.backbone_cfgs = cfgs['CSPDARKNET']
        self.neck_cfgs = cfgs['PAFPN']
        self.head_cfgs = cfgs['YOLOXHEAD']
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
        stride = [8, 16, 32]
        # loss parameters
        self.use_l1 = True
        # evaluate parameters
        self.nms_threshold = 0.7
        self.confidence_threshold = 0.1
        # dataloader
        self.img_size_train = tuple(self.dataset_cfgs['TRAIN_SIZE'])
        self.img_size_val = tuple(self.dataset_cfgs['VAL_SIZE'])
        # validation
        self.detect_list = []
        self.image_id_list = []
        self.origin_hw_list = []

        self.backbone = BACKBONE.CSPDarkNet(b_depth, b_channels, out_features, b_norm, b_act)
        self.neck = NECK.PAFPN(n_depth, out_features, n_channels, n_norm, n_act)
        self.head = HEAD.YOLOXHead(self.num_classes, stride, n_channels, n_norm, n_act)
        self.decoder = HEAD.YOLOXDecoder(self.num_classes, stride, n_channels, n_norm, n_act)

    def forward(self, x):
        x = self.backbone(x)
        x = self.neck(x)
        print(x.size())
        return x

    def training_step(self, batch, batch_idx):
        imgs, labels, _, _, _ = batch
        output = self.backbone(imgs)
        output = self.neck(output)
        pred, x_shifts, y_shifts, expand_strides = self.decoder(output)
        loss, iou_loss, conf_loss, cls_loss, l1_loss, num_fg = HEAD.YOLOXLoss(
            labels, pred, x_shifts, y_shifts, expand_strides, self.num_classes, self.use_l1)
        # loss, iou_loss, conf_loss, cls_loss, l1_loss, num_fg = self.head(output, labels, imgs)
        output = {
            "total_loss": loss,
            "iou_loss": iou_loss,
            "l1_loss": l1_loss,
            "conf_loss": conf_loss,
            "cls_loss": cls_loss,
            "num_fg": num_fg,
        }
        self.log("iou_loss", iou_loss, prog_bar=True)
        self.log("l1_loss", l1_loss, prog_bar=True)
        self.log("conf_loss", conf_loss, prog_bar=True)
        self.log("cls_loss", cls_loss, prog_bar=True)
        self.log("num_fg", num_fg, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        imgs, labels, img_hw, image_id, img_name = batch
        output = self.backbone(imgs)
        output = self.neck(output)
        pred, _, _, _ = self.decoder(output)
        detections = postprocess(pred.cpu(), self.num_classes, self.nms_threshold, self.confidence_threshold)
        self.detect_list.append(detections)
        self.image_id_list.append(image_id)
        self.origin_hw_list.append(img_hw)

    def validation_epoch_end(self, *args, **kwargs):
        data_dir = self.dataset_cfgs['DIR']
        json = self.dataset_cfgs['VAL_JSON']
        COCOEvaluator(self.detect_list, self.image_id_list, self.origin_hw_list, self.img_size_val, data_dir, json)
        self.detect_list = []
        self.image_id_list = []
        self.origin_hw_list = []

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=0.03)
        return optimizer

    def train_dataloader(self):
        from data.datasets.coco import COCODataset
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
                flip_prob=0.5,
                hsv_prob=1.0
            )
        )
        train_loader = DataLoader(dataset, batch_size=16, num_workers=4, shuffle=False)
        return train_loader

    def val_dataloader(self):
        from data.datasets.coco import COCODataset
        from torch.utils.data.dataloader import DataLoader
        data_dir = self.dataset_cfgs['DIR']
        json = self.dataset_cfgs['VAL_JSON']
        dataset = COCODataset(
            data_dir,
            json,
            name="val",
            img_size=self.img_size_val,
            preprocess=ValTransform(legacy=False),
        )
        val_loader = DataLoader(dataset, batch_size=16, num_workers=4, shuffle=False)
        return val_loader


