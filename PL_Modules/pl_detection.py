import time
import torch
from pytorch_lightning import LightningModule
# Train
from models.utils.ema import ModelEMA
from torch.optim import SGD
from models.layers.lr_scheduler import CosineWarmupScheduler
# Evaluate
from utils.flops import model_summary
from models.evaluators.postprocess import postprocess, format_outputs
from models.evaluators.eval_coco import COCOEvaluator
from models.evaluators.eval_voc import VOCEvaluator


class LitDetection(LightningModule):

    def __init__(self, model, model_cfgs, data_cfgs):
        super().__init__()
        self.model = model
        # Parameters for training
        self.co = model_cfgs['optimizer']
        self.cd = data_cfgs['dataset']
        self.nms_threshold = 0.65
        self.confidence_threshold = 0.01
        self.num_classes = data_cfgs['num_classes']
        self.img_size_train = tuple(self.cd['train_size'])
        self.img_size_val = tuple(self.cd['val_size'])
        # Training
        self.ema = self.co['ema']
        self.warmup = self.co['warmup']
        self.infr_times = []
        self.nms_times = []

        self.automatic_optimization = False
        self.ema_model = None
        self.ap50_95 = 0
        self.ap50 = 0

    def on_train_start(self) -> None:
        if self.ema is True:
            self.ema_model = ModelEMA(self.model, 0.9998)
        model_summary(self.model, self.img_size_train, self.device)

    def training_step(self, batch, batch_idx):
        imgs, labels, _, _, _ = batch
        losses = self.model(imgs, labels)
        self.log_dict(losses)
        self.log("total_loss", losses['loss'], prog_bar=True)
        self.log("lr", self.trainer.optimizers[0].param_groups[0]['lr'], prog_bar=True)
        # Backward
        optimizer = self.optimizers()
        optimizer.zero_grad()
        self.manual_backward(losses['loss'])
        optimizer.step()
        if self.ema is True:
            self.ema_model.update(self.model)
        self.lr_schedulers().step()

    def validation_step(self, batch, batch_idx):
        imgs, labels, img_hw, image_id, img_name = batch
        if self.ema_model is not None:
            model = self.ema_model.ema
        else:
            model = self.model
        start_time = time.time()
        detections = model(imgs, labels)
        self.infr_times.append(time.time() - start_time)
        start_time = time.time()
        detections = postprocess(detections, self.confidence_threshold, self.nms_threshold)
        self.nms_times.append(time.time() - start_time)
        json_det, det = format_outputs(detections, image_id, img_hw, self.img_size_val,
                                       self.trainer.datamodule.dataset_val.class_ids, labels)
        return json_det, det

    def validation_epoch_end(self, val_step_outputs):
        json_list = []
        det_list = []
        for i in range(len(val_step_outputs)):
            json_list += val_step_outputs[i][0]
            det_list += val_step_outputs[i][1]
        # COCO Evaluator
        ap50_95, ap50, summary = COCOEvaluator(json_list, self.trainer.datamodule.dataset_val)
        print("Batch {:d}, mAP = {:.3f}, mAP50 = {:.3f}".format(self.current_epoch, ap50_95, ap50))
        print(summary)
        # VOC Evaluator
        VOCEvaluator(det_list, self.trainer.datamodule.dataset_val, iou_thr=0.5)

        self.log("mAP", ap50_95, prog_bar=False)
        self.log("mAP50", ap50, prog_bar=False)
        if ap50_95 > self.ap50_95:
            self.ap50_95 = ap50_95
        if ap50 > self.ap50:
            self.ap50 = ap50

        average_ifer_time = torch.tensor(self.infr_times, dtype=torch.float32).mean().item()
        average_nms_time = torch.tensor(self.nms_times, dtype=torch.float32).mean().item()
        print("The average iference time is %.4fs, nms time is %.4fs" % (average_ifer_time, average_nms_time))
        self.infr_times, self.nms_times = [], []

    def configure_optimizers(self):
        optimizer = SGD(self.parameters(), lr=self.co["learning_rate"], momentum=self.co["momentum"])
        total_steps = self.trainer.estimated_stepping_batches
        lr_scheduler = CosineWarmupScheduler(optimizer, warmup=self.warmup * total_steps, max_iters=total_steps)
        return [optimizer], [lr_scheduler]

    def on_train_end(self) -> None:
        print("Best mAP = {:.3f}, best mAP50 = {:.3f}".format(self.ap50_95, self.ap50))

    def forward(self, imgs):
        self.model.eval()
        detections = self.model(imgs)
        return detections
