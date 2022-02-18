import torch
import torch.nn as nn
import numpy as np
import math


class YOLOLoss(nn.Module):
    def __init__(self, anchors, num_classes, img_size):
        super(YOLOLoss, self).__init__()
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

    def forward(self, input, targets=None):
        bs = input.size(0)
        in_h = input.size(2)
        in_w = input.size(3)
        stride_h = self.img_size[1] / in_h
        stride_w = self.img_size[0] / in_w
        scaled_anchors = [(a_w / stride_w, a_h / stride_h) for a_w, a_h in self.anchors]

        prediction = input.view(bs,  self.num_anchors,
                                self.bbox_attrs, in_h, in_w).permute(0, 1, 3, 4, 2).contiguous()

        # Get outputs
        x = torch.sigmoid(prediction[..., 0])  # Center x
        y = torch.sigmoid(prediction[..., 1])  # Center y
        w = prediction[..., 2]  # Width
        h = prediction[..., 3]  # Height
        conf = torch.sigmoid(prediction[..., 4])  # Conf
        pred_cls = torch.sigmoid(prediction[..., 5:])  # Cls pred.

        #  build target
        # From [cx, cy, w, h] to [x1%, y1%, x2%, y2%]
        nlabel = (targets.sum(dim=2) > 0).sum(dim=1)
        for batch_idx in range(bs):
            num_gt = int(nlabel[batch_idx])
            # x1 = (targets[batch_idx, :num_gt, 1] - targets[batch_idx, :num_gt, 3] / 2) / self.img_size[0]
            # x2 = (targets[batch_idx, :num_gt, 1] + targets[batch_idx, :num_gt, 3] / 2) / self.img_size[0]
            # y1 = (targets[batch_idx, :num_gt, 2] - targets[batch_idx, :num_gt, 4] / 2) / self.img_size[1]
            # y2 = (targets[batch_idx, :num_gt, 2] + targets[batch_idx, :num_gt, 4] / 2) / self.img_size[1]
            percent_cx = targets[batch_idx, :num_gt, 1] / self.img_size[0]
            percent_cy = targets[batch_idx, :num_gt, 2] / self.img_size[1]
            percent_w = targets[batch_idx, :num_gt, 3] / self.img_size[0]
            percent_y = targets[batch_idx, :num_gt, 4] / self.img_size[1]
            targets[batch_idx, :num_gt, 1] = percent_cx
            targets[batch_idx, :num_gt, 2] = percent_cy
            targets[batch_idx, :num_gt, 3] = percent_w
            targets[batch_idx, :num_gt, 4] = percent_y



        mask, noobj_mask, tx, ty, tw, th, tconf, tcls = self.get_target(targets, scaled_anchors,
                                                                        in_w, in_h, self.ignore_threshold)
        # mask, noobj_mask = mask.cuda(), noobj_mask.cuda()
        # tx, ty, tw, th = tx.cuda(), ty.cuda(), tw.cuda(), th.cuda()
        # tconf, tcls = tconf.cuda(), tcls.cuda()
        #  losses
        loss_x = self.bce_loss(x * mask, tx * mask)
        loss_y = self.bce_loss(y * mask, ty * mask)
        loss_w = self.mse_loss(w * mask, tw * mask)
        loss_h = self.mse_loss(h * mask, th * mask)
        loss_conf = self.bce_loss(conf * mask, mask) + \
                    0.5 * self.bce_loss(conf * noobj_mask, noobj_mask * 0.0)
        loss_cls = self.bce_loss(pred_cls[mask == 1], tcls[mask == 1])
        #  total loss = losses * weight
        loss = loss_x * self.lambda_xy + loss_y * self.lambda_xy + \
               loss_w * self.lambda_wh + loss_h * self.lambda_wh + \
               loss_conf * self.lambda_conf + loss_cls * self.lambda_cls

        # return loss, loss_x.item(), loss_y.item(), loss_w.item(), \
        #        loss_h.item(), loss_conf.item(), loss_cls.item()
        return loss

    def get_target(self, target, anchors, in_w, in_h, ignore_threshold):
        bs = target.size(0)

        mask = torch.zeros(bs, self.num_anchors, in_h, in_w, requires_grad=False)
        noobj_mask = torch.ones(bs, self.num_anchors, in_h, in_w, requires_grad=False)
        tx = torch.zeros(bs, self.num_anchors, in_h, in_w, requires_grad=False)
        ty = torch.zeros(bs, self.num_anchors, in_h, in_w, requires_grad=False)
        tw = torch.zeros(bs, self.num_anchors, in_h, in_w, requires_grad=False)
        th = torch.zeros(bs, self.num_anchors, in_h, in_w, requires_grad=False)
        tconf = torch.zeros(bs, self.num_anchors, in_h, in_w, requires_grad=False)
        tcls = torch.zeros(bs, self.num_anchors, in_h, in_w, self.num_classes, requires_grad=False)
        for b in range(bs):
            for t in range(target.shape[1]):
                if target[b, t].sum() == 0:
                    continue
                # Convert to position relative to box
                gx = target[b, t, 1] * in_w
                gy = target[b, t, 2] * in_h
                gw = target[b, t, 3] * in_w
                gh = target[b, t, 4] * in_h
                # Get grid box indices
                gi = int(gx)
                gj = int(gy)
                # Get shape of gt box
                gt_box = torch.FloatTensor(np.array([0, 0, gw, gh])).unsqueeze(0)
                # Get shape of anchor box
                anchor_shapes = torch.FloatTensor(np.concatenate((np.zeros((self.num_anchors, 2)),
                                                                  np.array(anchors)), 1))
                # Calculate iou between gt and anchor shapes
                anch_ious = bbox_iou(gt_box, anchor_shapes)
                # Where the overlap is larger than threshold set mask to zero (ignore)
                noobj_mask[b, anch_ious > ignore_threshold, gj, gi] = 0
                # Find the best matching anchor box
                best_n = np.argmax(anch_ious)

                # Masks
                mask[b, best_n, gj, gi] = 1
                # Coordinates
                tx[b, best_n, gj, gi] = gx - gi
                ty[b, best_n, gj, gi] = gy - gj
                # Width and height
                tw[b, best_n, gj, gi] = math.log(gw/anchors[best_n][0] + 1e-16)
                th[b, best_n, gj, gi] = math.log(gh/anchors[best_n][1] + 1e-16)
                # object
                tconf[b, best_n, gj, gi] = 1
                # One-hot encoding of label
                tcls[b, best_n, gj, gi, int(target[b, t, 0])] = 1

        return mask, noobj_mask, tx, ty, tw, th, tconf, tcls


def YOLOv3Loss(labels, outputs, num_classes, anchors, strides, maps_h, maps_w, thresh=0.5):
    preds_bbox = outputs[..., :4]  # [batch, n_anchors_all, 4]
    preds_obj = outputs[..., 4]  # [batch, n_anchors_all]
    preds_cls = outputs[..., 5:]  # [batch, n_anchors_all, n_classes]

    target_coord = torch.zeros_like(preds_bbox)
    target_obj = torch.zeros_like(preds_obj)
    target_cls = torch.zeros_like(preds_cls)

    mask_conf_pos = torch.zeros_like(preds_obj)
    mask_conf_neg = torch.ones_like(preds_obj)
    mask_coord = torch.zeros_like(preds_obj)
    mask_cls = torch.zeros_like(preds_obj)


    nlabel = (labels.sum(dim=2) > 0).sum(dim=1)  # number of objects in each image [batch]
    total_num_anchors = outputs.shape[1]  # n_anchors_all (int)
    anchors = torch.tensor(anchors)
    anchors_num = anchors.shape[0]
    anchors_box = torch.cat([torch.zeros_like(anchors), anchors], 1)  # [anchors_num, 4]

    fg_masks = []
    iou_sum = 0
    num_fgs = 0
    num_gt_batch = 0

    for batch_idx in range(outputs.shape[0]):
        num_gt = int(nlabel[batch_idx])
        num_gt_batch += num_gt
        if num_gt == 0:
            target_cls = outputs.new_zeros((0, num_classes))
            target_reg = outputs.new_zeros((0, 4))
            target_l1 = outputs.new_zeros((0, 4))
            target_obj = outputs.new_zeros((total_num_anchors, 1))
            fg_mask = outputs.new_zeros(total_num_anchors).bool()
        else:
            gt_bboxes_per_image = labels[batch_idx, :num_gt, 1:5]  # [n_gt, 4]
            gt_classes_per_image = labels[batch_idx, :num_gt, 0]  # [n_gt]
            preds_bboxes_per_image = preds_bbox[batch_idx]  # [n_anchor, 4]
            preds_obj_per_image = preds_obj[batch_idx]  # [n_anchor, 1]
            preds_cls_per_image = preds_cls[batch_idx]  # [n_anchor, n_class]

            # Select from all predictions
            iou_gt_pred = bbox_ious(gt_bboxes_per_image, preds_bboxes_per_image)  # [1,n_anchor]
            mask_iou = (iou_gt_pred > thresh).sum(0) >= 1   # [n_anchor]

            # Select from anchors
            gt_wh = gt_bboxes_per_image.clone()
            gt_wh[:, :2] = 0
            iou_gt_anchors = bbox_ious(gt_wh, anchors_box)
            _, best_anchors = iou_gt_anchors.max(1)

            for i in range(num_gt):
                gi = min(gt_bboxes_per_image[i, 2] - 1, max(0, int(gt_bboxes_per_image[i, 0])))
                gj = min(gt_bboxes_per_image[i, 3] - 1, max(0, int(gt_bboxes_per_image[i, 1])))
                curs = []
                the_grids = []
                stack = 0
                for j in range(len(maps_h)):
                    cur_w = int(gt_bboxes_per_image[i, 0] / strides[j])
                    cur_h = int(gt_bboxes_per_image[i, 1] / strides[j])
                    cur = (cur_h * maps_h[j] + cur_w) * anchors_num + best_anchors[i] + stack
                    curs.append(cur)
                    stack += maps_h[j] * maps_w[j] * anchors_num
                    iou = iou_gt_pred[:, cur]
                    iou_sum += iou
                    mask_conf_pos[batch_idx, cur] = 1
                    mask_conf_neg[batch_idx, cur] = 0
                    mask_cls[batch_idx, cur] = 1
                    target_coord[batch_idx, cur, :3] = gt_bboxes_per_image[i, :3]
                    target_obj[batch_idx, cur] = 1
                    target_cls[batch_idx, cur] = gt_classes_per_image[i]


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

    # get the coordinates of the intersection rectangle
    inter_rect_x1 =  torch.max(b1_x1, b2_x1)
    inter_rect_y1 =  torch.max(b1_y1, b2_y1)
    inter_rect_x2 =  torch.min(b1_x2, b2_x2)
    inter_rect_y2 =  torch.min(b1_y2, b2_y2)
    # Intersection area
    inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1 + 1, min=0) * \
        torch.clamp(inter_rect_y2 - inter_rect_y1 + 1, min=0)
    # Union Area
    b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)

    iou = inter_area / (b1_area + b2_area - inter_area + 1e-16)

    return iou







def bbox_ious(boxes1, boxes2):
    """ Compute IOU between all boxes from ``boxes1`` with all boxes from ``boxes2``.

    Args:
        boxes1 (torch.Tensor): List of bounding boxes
        boxes2 (torch.Tensor): List of bounding boxes

    Note:
        List format: [[xc, yc, w, h],...]
    """
    b1_len = boxes1.size(0)
    b2_len = boxes2.size(0)

    b1x1, b1y1 = (boxes1[:, :2] - (boxes1[:, 2:4] / 2)).split(1, 1)
    b1x2, b1y2 = (boxes1[:, :2] + (boxes1[:, 2:4] / 2)).split(1, 1)
    b2x1, b2y1 = (boxes2[:, :2] - (boxes2[:, 2:4] / 2)).split(1, 1)
    b2x2, b2y2 = (boxes2[:, :2] + (boxes2[:, 2:4] / 2)).split(1, 1)

    dx = (b1x2.min(b2x2.t()) - b1x1.max(b2x1.t())).clamp(min=0)
    dy = (b1y2.min(b2y2.t()) - b1y1.max(b2y1.t())).clamp(min=0)
    intersections = dx * dy

    areas1 = (b1x2 - b1x1) * (b1y2 - b1y1)
    areas2 = (b2x2 - b2x1) * (b2y2 - b2y1)
    unions = (areas1 + areas2.t()) - intersections

    return intersections / unions
