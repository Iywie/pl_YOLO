import torch
import torch.nn as nn
import torchvision


class YOLOXDecoder:
    def __init__(
            self,
            num_classes,
            strides=None,
    ):
        super().__init__()
        self.n_anchors = 1
        self.num_classes = num_classes
        self.strides = strides
        self.grids = [torch.zeros(1)] * len(strides)
        self.max_det = 300  # maximum number of detections per image
        self.max_nms = 10000  # maximum number of boxes into torchvision.ops.nms()

    def __call__(self, inputs, conf_thre=0.7, nms_thre=0.45, class_agnostic=False):
        preds = []
        batch_size = inputs[0].shape[0]
        n_ch = 4 + 1 + self.n_anchors * self.num_classes  # the channel of one ground truth prediction.

        for i in range(len(inputs)):
            # Change all inputs to the same channel.
            pred = inputs[i]
            h, w = pred.shape[-2:]

            # Three steps to localize predictions: grid, shifts of x and y, grid with stride
            if self.grids[i].shape[2:4] != pred.shape[2:4]:
                xv, yv = torch.meshgrid([torch.arange(h), torch.arange(w)], indexing='xy')
                grid = torch.stack((xv, yv), 2).view(1, 1, h, w, 2).type_as(pred)
                grid = grid.view(1, -1, 2)
                # grid: [1, h * w, 2]
                self.grids[i] = grid
            else:
                grid = self.grids[i]

            pred = pred.view(batch_size, self.n_anchors, n_ch, h, w)
            pred = pred.permute(0, 1, 3, 4, 2).reshape(
                batch_size, self.n_anchors * h * w, -1
            )
            # pred: [batch_size, h * w, n_ch]
            pred[..., :2] = (pred[..., :2] + grid) * self.strides[i]
            # The predictions of w and y are logs
            pred[..., 2:4] = torch.exp(pred[..., 2:4]) * self.strides[i]
            preds.append(pred)

        # preds: [batch_size, all predictions, n_ch]
        predictions = torch.cat(preds, 1)
        predictions[..., 4] = predictions[..., 4].sigmoid()
        predictions[..., 5:] = predictions[..., 5:].sigmoid()

        # from (cx,cy,w,h) to (x1,y1,x2,y2)
        box_corner = predictions.new(predictions.shape)
        box_corner[:, :, 0] = predictions[:, :, 0] - predictions[:, :, 2] / 2
        box_corner[:, :, 1] = predictions[:, :, 1] - predictions[:, :, 3] / 2
        box_corner[:, :, 2] = predictions[:, :, 0] + predictions[:, :, 2] / 2
        box_corner[:, :, 3] = predictions[:, :, 1] + predictions[:, :, 3] / 2
        predictions[:, :, :4] = box_corner[:, :, :4]

        output = [None for _ in range(len(predictions))]
        for i, image_pred in enumerate(predictions):

            # If none are remaining => process next image
            if not image_pred.size(0):
                continue
            # Get score and class with highest confidence
            class_conf, class_pred = torch.max(image_pred[:, 5: 5 + self.num_classes], 1, keepdim=True)

            conf_mask = (image_pred[:, 4] * class_conf.squeeze() >= conf_thre).squeeze()
            # Detections ordered as (x1, y1, x2, y2, obj_conf, class_conf, class_pred)
            detections = torch.cat((image_pred[:, :5], class_conf, class_pred.float()), 1)
            detections = detections[conf_mask]
            if detections.shape[0] > self.max_nms:
                detections = detections[:self.max_nms]
            if not detections.size(0):
                continue

            if class_agnostic:
                nms_out_index = torchvision.ops.nms(
                    detections[:, :4],
                    detections[:, 4] * detections[:, 5],
                    nms_thre,
                )
            else:
                nms_out_index = torchvision.ops.batched_nms(
                    detections[:, :4],
                    detections[:, 4] * detections[:, 5],
                    detections[:, 6],
                    nms_thre,
                )

            detections = detections[nms_out_index]
            if detections.shape[0] > self.max_det:  # limit detections
                detections = detections[:self.max_det]
            # output[i] = detections
            if output[i] is None:
                output[i] = detections
            else:
                output[i] = torch.cat((output[i], detections))

        return output

