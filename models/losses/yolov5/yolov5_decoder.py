import time
import torch
import torchvision
from models.utils.bbox import xywh2xyxy, box_iou


class YOLOv5Decoder:
    def __init__(self, num_classes, anchors, strides):
        super(YOLOv5Decoder, self).__init__()
        self.num_classes = num_classes
        self.anchors = torch.tensor(anchors)
        self.strides = strides
        self.grid = [torch.zeros(1)] * len(anchors)  # init grid
        self.anchor_grid = self.anchors.clone().view(len(anchors), 1, -1, 1, 1, 2)

    def __call__(self, inputs, conf_thre=0.7, nms_thre=0.45, multi_label=False, agnostic=False):
        """
        :param inputs: a list of feature maps
        :return:
        """
        predictions = self.decode(inputs)

        obj_mask = predictions[..., 4] > conf_thre  # candidates

        # NMS Settings
        min_wh, max_wh = 2, 4096  # (pixels) minimum and maximum box width and height
        max_det = 300  # maximum number of detections per image
        max_nms = 30000  # maximum number of boxes into torchvision.ops.nms()
        time_limit = 10.0  # seconds to quit after
        redundant = True  # require redundant detections
        multi_label &= self.num_classes > 1  # multiple labels per box (adds 0.5ms/img)
        merge = False  # use merge-NMS

        t = time.time()
        output = [torch.zeros((0, 7), device=predictions.device)] * predictions.shape[0]
        for img_idx, x in enumerate(predictions):  # image index, image inference
            # Apply constraints
            # x[((x[..., 2:4] < min_wh) | (x[..., 2:4] > max_wh)).any(1), 4] = 0  # width-height
            x = x[obj_mask[img_idx]]  # is object

            # If none remain process next image
            if not x.shape[0]:
                continue

            # Box (center x, center y, width, height) to (x1, y1, x2, y2)
            box = xywh2xyxy(x[:, :4])
            x[:, :4] = box

            # Compute confidence
            pred_conf = x[:, 5:] * x[:, 4].unsqueeze(1)
            obj = x[:, 4]
            # Detections matrix nx6 (xyxy, conf, cls)
            if multi_label:
                # the coordinate of all class predictions
                i, j = (pred_conf > conf_thre).nonzero(as_tuple=False).T  # conf = obj_conf * cls_conf
                x = torch.cat((box[i], obj[i], x[i, j + 5, None], j[:, None].float()), 1)
            else:  # best class only
                conf, j = x[:, 5:].max(1, keepdim=True)
                conf_mask = ((obj[:, None] * conf) >= conf_thre).squeeze(-1)
                x = torch.cat((box, obj[:, None], conf, j.float()), 1)[conf_mask]

            # Check shape
            n = x.shape[0]  # number of boxes
            if not n:  # no boxes
                continue
            elif n > max_nms:  # excess boxes
                x = x[x[:, 4].argsort(descending=True)[:max_nms]]  # sort by confidence

            # Batched NMS
            c = x[:, 6] * (0 if agnostic else max_wh)  # classes
            boxes, scores = x[:, :4] + c.unsqueeze(-1), x[:, 4]  # boxes (offset by class), scores
            i = torchvision.ops.nms(boxes, scores, nms_thre)  # NMS
            if i.shape[0] > max_det:  # limit detections
                i = i[:max_det]

            if merge and (1 < n < 3E3):  # Merge NMS (boxes merged using weighted mean)
                # update boxes as boxes(i,4) = weights(i,n) * boxes(n,4)
                iou = box_iou(boxes[i], boxes) > nms_thre  # iou matrix
                weights = iou * scores[None]  # box weights
                x[i, :4] = torch.mm(weights, x[:, :4]).float() / weights.sum(1, keepdim=True)  # merged boxes
                if redundant:
                    i = i[iou.sum(1) > 1]  # require redundancy

            output[img_idx] = x[i]
            if (time.time() - t) > time_limit:
                print(f'WARNING: NMS time limit {time_limit}s exceeded')
                break  # time limit exceeded

        return output

    def decode(self, inputs):
        nl = len(inputs)  # number of layers
        na = len(self.anchors[0])  # number of anchors, every layer have the same number of anchors
        bs = inputs[0].shape[0]
        z = []  # inference output
        for i in range(nl):
            ny = inputs[i].size(2)
            nx = inputs[i].size(3)
            prediction = inputs[i].view(bs, na, 5 + self.num_classes, ny, nx) \
                .permute(0, 1, 3, 4, 2).contiguous()
            inputs[i] = prediction
            if self.grid[i].shape[2:4] != prediction.shape[2:4]:
                self.grid[i] = self._make_grid(nx, ny).type_as(prediction)
                self.anchor_grid = self.anchor_grid.type_as(prediction)
            y = prediction.sigmoid()
            y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + self.grid[i]) * self.strides[i]  # xy
            y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
            z.append(y.view(bs, -1, 5 + self.num_classes))

        return torch.cat(z, 1)

    @staticmethod
    def _make_grid(nx=20, ny=20):
        xv, yv = torch.meshgrid([torch.arange(ny), torch.arange(nx)], indexing='xy')
        return torch.stack((xv, yv), 2).view((1, 1, ny, nx, 2)).float()
