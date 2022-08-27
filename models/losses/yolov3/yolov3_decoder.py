import torch
import time
import torch.nn as nn
import torchvision


class YOLOv3Decoder:
    def __init__(self, num_classes, anchors, strides):
        super().__init__()
        self.num_classes = num_classes
        self.anchors = torch.Tensor(anchors)
        self.strides = strides
        self.grids = [torch.zeros(1)] * len(strides)
        self.n_ch = 4 + 1 + self.num_classes
        self.max_det = 300  # maximum number of detections per image
        self.max_nms = 10000  # maximum number of boxes into torchvision.ops.nms()
        # self.multi_label = self.num_classes > 1  # multiple labels per box (adds 0.5ms/img)
        self.multi_label = False
        self.max_wh = 4096
        self.time_limit = 1.0  # seconds to quit after

    def __call__(self, inputs, conf_thre=0.7, nms_thre=0.45, classes=None):
        preds = []
        for i in range(len(inputs)):
            batch_size = inputs[i].size(0)
            map_h = inputs[i].size(2)
            map_w = inputs[i].size(3)
            stride = self.strides[i]
            anchors = self.anchors[i]
            num_anchors = len(anchors)

            scaled_anchors = [(a_w / stride, a_h / stride) for a_w, a_h in anchors]

            # prediction: [batch_size, num_anchors, map_h, map_w, bbox_attrs]
            pred = inputs[i].view(batch_size, num_anchors, self.n_ch, map_h, map_w) \
                .permute(0, 1, 3, 4, 2).contiguous()

            if self.grids[i].shape[2:4] != pred.shape[2:4]:
                xv, yv = torch.meshgrid([torch.arange(map_h), torch.arange(map_w)], indexing='xy')
                grid = torch.stack((xv, yv), 2).view(1, 1, map_h, map_w, 2).type_as(pred)
                self.grids[i] = grid
            else:
                grid = self.grids[i]

            # Calculate anchor w, h
            anchor_w = torch.tensor(scaled_anchors).index_select(1, torch.tensor([0])).type_as(pred)
            anchor_h = torch.tensor(scaled_anchors).index_select(1, torch.tensor([1])).type_as(pred)
            anchor_w = anchor_w.repeat(batch_size, 1, map_h * map_w).view(batch_size, -1, map_h, map_w)
            anchor_h = anchor_h.repeat(batch_size, 1, map_h * map_w).view(batch_size, -1, map_h, map_w)

            # pred: [batch_size, h * w, n_ch]
            pred[..., :2] = (pred[..., :2].sigmoid() + grid) * self.strides[i]
            # The predictions of w and y are logs
            pred[..., 2] = torch.exp(pred[..., 2]) * anchor_w * self.strides[i]
            pred[..., 3] = torch.exp(pred[..., 3]) * anchor_h * self.strides[i]
            pred[..., 4] = pred[..., 4].sigmoid()
            pred[..., 5:] = pred[..., 5:].sigmoid()
            pred = pred.view(batch_size, -1, self.n_ch)
            preds.append(pred)

        # preds: [batch_size, all predictions, n_ch]
        predictions = torch.cat(preds, 1)

        # from (cx,cy,w,h) to (x1,y1,x2,y2)
        box_corner = predictions.new(predictions.shape)
        box_corner[:, :, 0] = predictions[:, :, 0] - predictions[:, :, 2] / 2
        box_corner[:, :, 1] = predictions[:, :, 1] - predictions[:, :, 3] / 2
        box_corner[:, :, 2] = predictions[:, :, 0] + predictions[:, :, 2] / 2
        box_corner[:, :, 3] = predictions[:, :, 1] + predictions[:, :, 3] / 2
        predictions[:, :, :4] = box_corner[:, :, :4]

        t = time.time()

        output = [None for _ in range(len(predictions))]
        for b_idx, image_pred in enumerate(predictions):
            image_pred = image_pred[image_pred[..., 4] > conf_thre]  # confidence
            # If none are remaining => process next image
            if not image_pred.size(0):
                continue
            # Compute conf
            image_pred[:, 5:] *= image_pred[:, 4:5]  # conf = obj_conf * cls_conf
            # Detections matrix nx6 (xyxy, conf, cls)
            if self.multi_label:
                i, j = (image_pred[:, 5:] > conf_thre).nonzero(as_tuple=False).T
                x = torch.cat((image_pred[:, :5][i], image_pred[i, j + 5, None], j[:, None].float()), 1)
            else:  # best class only
                conf, j = image_pred[:, 5:].max(1, keepdim=True)
                x = torch.cat((image_pred[:, :5], conf, j.float()), 1)[conf.view(-1) > conf_thre]

            # Filter by class
            if classes is not None:
                x = x[(x[:, 6] == torch.tensor(classes, device=x.device)).any(1)]

            # Check shape
            n = x.shape[0]  # number of boxes
            if not n:  # no boxes
                continue
            elif n > self.max_nms:  # excess boxes
                # sort by confidence
                x = x[x[:, 5].argsort(descending=True)[:self.max_nms]]

            # Batched NMS
            c = x[:, 6:7] * self.max_wh  # classes
            # boxes (offset by class), scores
            boxes, scores = x[:, :4], x[:, 5]
            nms = torchvision.ops.nms(boxes, scores, nms_thre)  # NMS
            if nms.shape[0] > self.max_det:  # limit detections
                nms = nms[:self.max_det]

            output[b_idx] = x[nms]

            if (time.time() - t) > self.time_limit:
                print(f'WARNING: NMS time limit {self.time_limit}s exceeded')
                break  # time limit exceeded

        return output


def bbox_iou(box1, box2, x1y1x2y2=True):
    """
    Returns the IoU of two bounding boxes
    """
    if not x1y1x2y2:
        # Transform from center and width to exact coordinates
        b1_x1, b1_x2 = box1[:, 0] - box1[:, 2] / 2, box1[:, 0] + box1[:, 2] / 2
        b1_y1, b1_y2 = box1[:, 1] - box1[:, 3] / 2, box1[:, 1] + box1[:, 3] / 2
        b2_x1, b2_x2 = box2[:, 0] - box2[:, 2] / 2, box2[:, 0] + box2[:, 2] / 2
        b2_y1, b2_y2 = box2[:, 1] - box2[:, 3] / 2, box2[:, 1] + box2[:, 3] / 2
    else:
        # Get the coordinates of bounding boxes
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[:,0], box1[:,1], box1[:,2], box1[:,3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[:,0], box2[:,1], box2[:,2], box2[:,3]

    # get the corrdinates of the intersection rectangle
    inter_rect_x1 =  torch.max(b1_x1, b2_x1)
    inter_rect_y1 =  torch.max(b1_y1, b2_y1)
    inter_rect_x2 =  torch.min(b1_x2, b2_x2)
    inter_rect_y2 =  torch.min(b1_y2, b2_y2)
    # Intersection area
    inter_area =    torch.clamp(inter_rect_x2 - inter_rect_x1 + 1, min=0) * \
                    torch.clamp(inter_rect_y2 - inter_rect_y1 + 1, min=0)
    # Union Area
    b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)

    iou = inter_area / (b1_area + b2_area - inter_area + 1e-16)

    return iou