from pytorch_lightning import LightningModule

from data import TrainTransform, ValTransform
import models.backbones as backbone
import models.necks as neck
import models.heads as head
import models.evaluators as evaluator

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
        num_classes = self.head_cfgs['CLASSES']
        stride = [8, 16, 32]

        self.backbone = backbone.CSPDarkNet(b_depth, b_channels, out_features, b_norm, b_act)
        self.neck = neck.PAFPN(n_depth, out_features, n_channels, n_norm, n_act)
        self.head = head.YOLOXHead(num_classes, stride, n_channels, n_norm, n_act)
        self.decoder = evaluator.decoder()

    def forward(self, x):
        x = self.backbone(x)
        x = self.neck(x)
        print(x.size())
        return x

    def training_step(self, batch, batch_idx):
        imgs, labels, _, _, _ = batch
        preds = self.backbone(imgs)
        preds = self.neck(preds)
        loss, iou_loss, conf_loss, cls_loss, l1_loss, num_fg = self.head(preds, labels, imgs)
        output = {
            "total_loss": loss,
            "iou_loss": iou_loss,
            "l1_loss": l1_loss,
            "conf_loss": conf_loss,
            "cls_loss": cls_loss,
            "num_fg": num_fg,
        }
        return loss

    def validation_step(self, batch, batch_idx):
        imgs, labels, img_hw, image_id, img_name = batch
        preds = self.backbone(imgs)
        preds = self.neck(preds)
        preds = self.head(preds)
        output = self.decoder(preds)




    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=0.03)
        return optimizer

    def train_dataloader(self):
        from data.datasets.coco import COCODataset
        from torch.utils.data.dataloader import DataLoader
        data_dir = self.dataset_cfgs['DIR']
        json = self.dataset_cfgs['TRAIN_JSON']
        img_size = tuple(self.dataset_cfgs['TRAIN_SIZE'])
        dataset = COCODataset(
            data_dir,
            json,
            name='train',
            img_size=img_size,
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
        img_size = tuple(self.dataset_cfgs['TRAIN_SIZE'])
        dataset = COCODataset(
            data_dir,
            json,
            name="val",
            img_size=img_size,
            preprocess=ValTransform(legacy=False),
        )
        val_loader = DataLoader(dataset, batch_size=16, num_workers=4, shuffle=False)
        return val_loader


