import torch
import torch.nn as nn
import numpy as np
import math


class YOLOv3Loss(nn.Module):
    def __init__(self, anchors, num_classes, img_size):
        super(YOLOv3Loss, self).__init__()
        self.anchors = anchors
        self.num_anchors = len(anchors)
        self.num_classes = num_classes
        self.bbox_attrs = 5 + num_classes
        self.img_size = img_size

        self.ignore_threshold = 0.5
        self.lambda_xy = 2.5
        self.lambda_wh = 2.5
        self.lambda_conf = 1.0
        self.lambda_cls = 1.0

        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCELoss()

    def forward(self, inputs, targets=None):

        batch_size = inputs.size(0)
        input_h = inputs.size(2)
        input_w = inputs.size(3)
        stride_h = self.img_size[1] / input_h
        stride_w = self.img_size[0] / input_w
        scaled_anchors = [(a_w / stride_w, a_h / stride_h) for a_w, a_h in self.anchors]

        # prediction: [batch_size, num_anchors, map_h, map_w, bbox_attrs]
        prediction = inputs.view(batch_size, self.num_anchors, self.bbox_attrs, input_h, input_w) \
            .permute(0, 1, 3, 4, 2).contiguous()
        cx = torch.sigmoid(prediction[..., 0])          # Center x
        cy = torch.sigmoid(prediction[..., 1])          # Center y
        w = prediction[..., 2]                          # Width
        h = prediction[..., 3]                          # Height
        conf = torch.sigmoid(prediction[..., 4])        # Conf
        pred_cls = torch.sigmoid(prediction[..., 5:])   # Cls pred

        if targets is not None:
            nlabel = (targets.sum(dim=2) > 0).sum(dim=1)
            # From [cx, cy, w, h] to [x1%, y1%, x2%, y2%]
            for batch_idx in range(batch_size):
                num_gt = int(nlabel[batch_idx])
                percent_cx = targets[batch_idx, :num_gt, 1] / self.img_size[0]
                percent_cy = targets[batch_idx, :num_gt, 2] / self.img_size[1]
                percent_w = targets[batch_idx, :num_gt, 3] / self.img_size[0]
                percent_y = targets[batch_idx, :num_gt, 4] / self.img_size[1]
                targets[batch_idx, :num_gt, 1] = percent_cx
                targets[batch_idx, :num_gt, 2] = percent_cy
                targets[batch_idx, :num_gt, 3] = percent_w
                targets[batch_idx, :num_gt, 4] = percent_y

            # mask
            mask = torch.zeros(batch_size, self.num_anchors, input_h, input_w, requires_grad=False).type_as(prediction)
            noobj_mask = torch.ones(batch_size, self.num_anchors, input_h, input_w, requires_grad=False).type_as(prediction)
            # target
            tx = torch.zeros(batch_size, self.num_anchors, input_h, input_w, requires_grad=False).type_as(prediction)
            ty = torch.zeros(batch_size, self.num_anchors, input_h, input_w, requires_grad=False).type_as(prediction)
            tw = torch.zeros(batch_size, self.num_anchors, input_h, input_w, requires_grad=False).type_as(prediction)
            th = torch.zeros(batch_size, self.num_anchors, input_h, input_w, requires_grad=False).type_as(prediction)
            tconf = torch.zeros(batch_size, self.num_anchors, input_h, input_w, requires_grad=False).type_as(prediction)
            tcls = torch.zeros(batch_size, self.num_anchors, input_h, input_w, self.num_classes, requires_grad=False)\
                .type_as(prediction)
            for batch_idx in range(batch_size):
                num_gt = int(nlabel[batch_idx])
                gt_bboxes_per_image = targets[batch_idx, :num_gt, 1:5]
                gt_classes_per_image = targets[batch_idx, :num_gt, 0]
                for gt_idx in range(num_gt):
                    # Convert to position relative to box
                    # 还原的是在该特征图上的坐标，与原坐标不同。
                    gx = gt_bboxes_per_image[gt_idx, 0] * input_w
                    gy = gt_bboxes_per_image[gt_idx, 1] * input_h
                    gw = gt_bboxes_per_image[gt_idx, 2] * input_w
                    gh = gt_bboxes_per_image[gt_idx, 3] * input_h
                    # Get grid box indices
                    gi = int(gx)
                    gj = int(gy)
                    # Get shape of gt box
                    gt_box = torch.FloatTensor(np.array([0, 0, gw.cpu(), gh.cpu()])).unsqueeze(0)
                    # Get shape of anchor box
                    anchor_shapes = torch.FloatTensor(
                        np.concatenate((np.zeros((self.num_anchors, 2)), np.array(scaled_anchors)), 1))
                    # Calculate iou between gt and anchor shapes
                    anch_ious = bbox_iou(gt_box, anchor_shapes)
                    # Where the overlap is larger than threshold set mask to zero (ignore)
                    noobj_mask[batch_idx, anch_ious > self.ignore_threshold, gj, gi] = 0
                    # Find the best matching anchor box
                    best_n = np.argmax(anch_ious)
                    # Masks
                    mask[batch_idx, best_n, gj, gi] = 1
                    # relative coordinates in the grid
                    tx[batch_idx, best_n, gj, gi] = gx - gi
                    ty[batch_idx, best_n, gj, gi] = gy - gj
                    # Width and height
                    # gw: gt在该特征图上的长宽，除以anchor的长宽
                    # gt与scaled_anchor的比再log，是神经网络所求！
                    tw[batch_idx, best_n, gj, gi] = math.log(gw / scaled_anchors[best_n][0] + 1e-16)
                    th[batch_idx, best_n, gj, gi] = math.log(gh / scaled_anchors[best_n][1] + 1e-16)
                    # object
                    tconf[batch_idx, best_n, gj, gi] = 1
                    # One-hot encoding of label
                    tcls[batch_idx, best_n, gj, gi, int(gt_classes_per_image[gt_idx])] = 1

            loss_x = self.bce_loss(cx * mask, tx * mask)
            loss_y = self.bce_loss(cy * mask, ty * mask)
            loss_w = self.mse_loss(w * mask, tw * mask)
            loss_h = self.mse_loss(h * mask, th * mask)
            loss_conf = self.bce_loss(conf * mask, mask) + 0.5 * self.bce_loss(conf * noobj_mask, noobj_mask * 0.0)
            loss_cls = self.bce_loss(pred_cls[mask == 1], tcls[mask == 1])

            loss = loss_x * self.lambda_xy + loss_y * self.lambda_xy + \
                   loss_w * self.lambda_wh + loss_h * self.lambda_wh + \
                   loss_conf * self.lambda_conf + loss_cls * self.lambda_cls

            return loss

        else:
            grid_x = torch.linspace(0, input_w - 1, input_w).repeat(input_w, 1).repeat(batch_size * self.num_anchors, 1,
                                                                                       1).view(cx.shape).type_as(cx)
            grid_y = torch.linspace(0, input_h - 1, input_h).repeat(input_h, 1).repeat(batch_size * self.num_anchors, 1,
                                                                                       1).view(cy.shape).type_as(cy)
            # Calculate anchor w, h
            anchor_w = torch.tensor(scaled_anchors).index_select(1, torch.tensor([0])).type_as(prediction)
            anchor_h = torch.tensor(scaled_anchors).index_select(1, torch.tensor([1])).type_as(prediction)
            anchor_w = anchor_w.repeat(batch_size, 1).repeat(1, 1, input_h * input_w).view(w.shape)
            anchor_h = anchor_h.repeat(batch_size, 1).repeat(1, 1, input_h * input_w).view(h.shape)

            cx = cx + grid_x
            cy = cy + grid_y
            w = torch.exp(w) * anchor_w
            h = torch.exp(h) * anchor_h
            pred_bboxes = torch.cat((cx, cy, w, h), -1)

            _scale = torch.Tensor([stride_w, stride_h] * 2).type_as(pred_bboxes)
            output = torch.cat((pred_bboxes.view(batch_size, -1, 4) * _scale, conf.view(batch_size, -1, 1),
                                pred_cls.view(batch_size, -1, self.num_classes)), -1)

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
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]

    # get the corrdinates of the intersection rectangle
    inter_rect_x1 = torch.max(b1_x1, b2_x1)
    inter_rect_y1 = torch.max(b1_y1, b2_y1)
    inter_rect_x2 = torch.min(b1_x2, b2_x2)
    inter_rect_y2 = torch.min(b1_y2, b2_y2)
    # Intersection area
    inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1 + 1, min=0) * \
                 torch.clamp(inter_rect_y2 - inter_rect_y1 + 1, min=0)
    # Union Area
    b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)

    iou = inter_area / (b1_area + b2_area - inter_area + 1e-16)

    return iou
