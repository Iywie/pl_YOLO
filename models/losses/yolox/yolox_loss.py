import torch
import torch.nn as nn
import torch.nn.functional as F
from models.layers.losses.iou_loss import bboxes_iou, IOUloss


class YOLOXLoss(nn.Module):
    def __init__(self, num_classes, strides, use_l1=False):
        super(YOLOXLoss, self).__init__()
        self.num_classes = num_classes
        self.strides = strides
        self.n_anchors = 1
        self.grids = [torch.zeros(1)] * len(strides)
        self.use_l1 = use_l1

        self.iou_loss = IOUloss(reduction="none", loss_type="giou")
        self.bcewithlog_loss = nn.BCEWithLogitsLoss(reduction="none")
        self.l1_loss = nn.L1Loss(reduction="none")

    def __call__(self, inputs, labels):

        preds, oriboxes, x_shifts, y_shifts, expanded_strides = self.decode(inputs)

        # inference
        if not self.training:
            preds[..., 4] = preds[..., 4].sigmoid()
            preds[..., 5:] = preds[..., 5:].sigmoid()
            # from (cx,cy,w,h) to (x1,y1,x2,y2)
            box_corner = preds.new(preds.shape)
            box_corner = box_corner[:, :, 0:4]
            box_corner[:, :, 0] = preds[:, :, 0] - preds[:, :, 2] / 2
            box_corner[:, :, 1] = preds[:, :, 1] - preds[:, :, 3] / 2
            box_corner[:, :, 2] = preds[:, :, 0] + preds[:, :, 2] / 2
            box_corner[:, :, 3] = preds[:, :, 1] + preds[:, :, 3] / 2
            preds[:, :, :4] = box_corner[:, :, :4]
            return preds

        # training
        bbox_preds = preds[:, :, :4]  # [batch, n_anchors_all, 4]
        obj_preds = preds[:, :, 4].unsqueeze(-1)  # [batch, n_anchors_all, 1]
        cls_preds = preds[:, :, 5:]  # [batch, n_anchors_all, n_classes]

        nlabel = (labels.sum(dim=2) > 0).sum(dim=1)  # number of objects (list)
        total_num_anchors = preds.shape[1]  # n_anchors_all (int)

        cls_targets = []
        reg_targets = []
        l1_targets = []
        obj_targets = []
        fg_masks = []
        num_fgs = 0
        num_gts = 0

        for batch_idx in range(preds.shape[0]):
            num_gt = int(nlabel[batch_idx])
            num_gts += num_gt
            if num_gt == 0:
                cls_target = preds.new_zeros((0, self.num_classes))
                reg_target = preds.new_zeros((0, 4))
                l1_target = preds.new_zeros((0, 4))
                obj_target = preds.new_zeros((total_num_anchors, 1))
                fg_mask = preds.new_zeros(total_num_anchors).bool()
            else:
                gt_bboxes_per_image = labels[batch_idx, :num_gt, 1:5]
                gt_classes_per_image = labels[batch_idx, :num_gt, 0]
                bboxes_preds_per_image = bbox_preds[batch_idx]

                with torch.no_grad():
                    # Get valuable grids according ground truth
                    fg_mask, in_boxes_and_center_mask = get_in_boxes_info(
                        gt_bboxes_per_image,
                        expanded_strides,
                        x_shifts,
                        y_shifts,
                        total_num_anchors,
                        num_gt,
                    )

                    bboxes_preds_per_image = bboxes_preds_per_image[fg_mask]
                    cls_preds_ = cls_preds[batch_idx][fg_mask]
                    obj_preds_ = obj_preds[batch_idx][fg_mask]
                    num_in_boxes_anchor = bboxes_preds_per_image.shape[0]

                    pair_wise_ious = bboxes_iou(gt_bboxes_per_image, bboxes_preds_per_image, False)
                    # 以e为底的log函数，iou趋近1为0，iou为0趋近18.42
                    pair_wise_ious_loss = -torch.log(pair_wise_ious + 1e-8)

                    gt_cls_per_image = (
                        F.one_hot(gt_classes_per_image.to(torch.int64), self.num_classes)
                        .float()
                        .unsqueeze(1)
                        .repeat(1, num_in_boxes_anchor, 1)
                    )
                    with torch.cuda.amp.autocast(enabled=False):
                        cls_preds_ = (
                                cls_preds_.float().unsqueeze(0).repeat(num_gt, 1, 1).sigmoid_()
                                * obj_preds_.unsqueeze(0).repeat(num_gt, 1, 1).sigmoid_()
                        )
                        pair_wise_cls_loss = F.binary_cross_entropy(
                            cls_preds_.sqrt_(), gt_cls_per_image, reduction="none"
                        ).sum(-1)
                        del cls_preds_

                    cost = (
                            pair_wise_cls_loss
                            + 3.0 * pair_wise_ious_loss
                            + 100000.0 * (~in_boxes_and_center_mask)
                    )

                    # Dynamic k methods: select the final predictions.
                    (
                        fg_mask,
                        num_fg,
                        matched_gt_inds,
                        gt_matched_classes,
                        pred_ious_this_matching,
                    ) = dynamic_k_matching(fg_mask, cost, pair_wise_ious, gt_classes_per_image, num_gt)
                    del pair_wise_cls_loss, cost, pair_wise_ious, pair_wise_ious_loss

                # predict anchors of the whole batch.
                num_fgs += num_fg

                cls_target = F.one_hot(
                    gt_matched_classes.to(torch.int64), self.num_classes
                ) * pred_ious_this_matching.unsqueeze(-1)
                obj_target = fg_mask.unsqueeze(-1)
                reg_target = gt_bboxes_per_image[matched_gt_inds]
                if self.use_l1:
                    l1_target = get_l1_type(preds.new_zeros((num_fg, 4)),
                                            gt_bboxes_per_image[matched_gt_inds],
                                            expanded_strides[0][fg_mask],
                                            x_shifts=x_shifts[0][fg_mask],
                                            y_shifts=y_shifts[0][fg_mask], )
            cls_targets.append(cls_target)
            reg_targets.append(reg_target)
            obj_targets.append(obj_target.type_as(reg_target))
            fg_masks.append(fg_mask)
            if self.use_l1:
                l1_targets.append(l1_target)

        # all predict anchors of this batch images
        cls_targets = torch.cat(cls_targets, 0)
        reg_targets = torch.cat(reg_targets, 0)
        obj_targets = torch.cat(obj_targets, 0)
        fg_masks = torch.cat(fg_masks, 0)
        if self.use_l1:
            l1_targets = torch.cat(l1_targets, 0)
        num_fgs = max(num_fgs, 1)

        loss_iou = (self.iou_loss(bbox_preds.view(-1, 4)[fg_masks], reg_targets)).sum() / num_fgs

        loss_obj = (self.bcewithlog_loss(obj_preds.view(-1, 1), obj_targets)).sum() / num_fgs

        loss_cls = (self.bcewithlog_loss(cls_preds.view(-1, self.num_classes)[fg_masks], cls_targets)).sum() / num_fgs

        # L1loss is the distance among the four property of a predicted box.
        if self.use_l1:
            loss_l1 = (self.l1_loss(oriboxes.view(-1, 4)[fg_masks], l1_targets)).sum() / num_fgs
        else:
            loss_l1 = 0.0

        reg_weight = 5.0
        loss = reg_weight * loss_iou + loss_obj + loss_cls + loss_l1

        loss_dict = {
            "loss": loss,
            "loss_iou": loss_iou,
            "loss_obj": loss_obj,
            "loss_cls": loss_cls,
            "loss_l1": loss_l1,
            "proportion": num_fgs / max(num_gts, 1),
        }
        return loss_dict

    def decode(self, inputs):
        """
        :param
            inputs: feature maps
        :return:
            all predictions: (batch_size, n_anchors * h * w, n_ch)
            ori_boxes: (batch_size, n_anchors * h * w, 4)
        """
        preds = []
        ori_boxes = []
        x_shifts = []
        y_shifts = []
        expanded_strides = []
        batch_size = inputs[0].shape[0]
        n_ch = self.n_anchors * (self.num_classes + 5)  # the channel of one ground truth prediction.

        for i in range(len(inputs)):
            # Change all inputs to the same channel.
            pred = inputs[i]
            h, w = pred.shape[-2:]

            # Three steps to localize predictions: grid, shifts of x and y, grid with stride
            if self.grids[i].shape[2:4] != pred.shape[2:4]:
                xv, yv = torch.meshgrid([torch.arange(h), torch.arange(w)], indexing='xy')
                grid = torch.stack((xv, yv), 2).view(1, 1, h, w, 2).type_as(pred)
                grid = grid.view(1, -1, 2)  # (1, h * w, 2)
                self.grids[i] = grid
            else:
                grid = self.grids[i]
            x_shifts.append(grid[:, :, 0])
            y_shifts.append(grid[:, :, 1])
            expanded_strides.append(
                torch.zeros(1, grid.shape[1]).fill_(self.strides[i]).type_as(pred)
            )

            pred = pred.view(batch_size, self.n_anchors, n_ch, h, w)
            pred = pred.permute(0, 1, 3, 4, 2).reshape(
                batch_size, self.n_anchors * h * w, -1
            )
            ori_box = pred[..., :4].clone()
            ori_boxes.append(ori_box)
            # pred: [batch_size, h * w, n_ch]
            pred[..., :2] = (pred[..., :2] + grid) * self.strides[i]
            # The predictions of w and y are logs
            pred[..., 2:4] = torch.exp(pred[..., 2:4]) * self.strides[i]
            preds.append(pred)

        # preds: [batch_size, all predictions, n_ch]
        preds = torch.cat(preds, 1)
        ori_boxes = torch.cat(ori_boxes, 1)
        x_shifts = torch.cat(x_shifts, 1)
        y_shifts = torch.cat(y_shifts, 1)
        expanded_strides = torch.cat(expanded_strides, 1)
        return preds, ori_boxes, x_shifts, y_shifts, expanded_strides


def get_in_boxes_info(
        gt_bboxes_per_image,
        expanded_strides,
        x_shifts,
        y_shifts,
        total_num_anchors,
        num_gt,
):
    expanded_strides_per_image = expanded_strides[0]
    x_shifts_per_image = x_shifts[0] * expanded_strides_per_image
    y_shifts_per_image = y_shifts[0] * expanded_strides_per_image
    x_centers_per_image = (
        (x_shifts_per_image + 0.5 * expanded_strides_per_image).unsqueeze(0).repeat(num_gt, 1)
    )  # [n_anchor] -> [n_gt, n_anchor]
    y_centers_per_image = (
        (y_shifts_per_image + 0.5 * expanded_strides_per_image).unsqueeze(0).repeat(num_gt, 1)
    )

    gt_bboxes_per_image_l = (
        (gt_bboxes_per_image[:, 0] - 0.5 * gt_bboxes_per_image[:, 2])
        .unsqueeze(1)
        .repeat(1, total_num_anchors)
    )
    gt_bboxes_per_image_r = (
        (gt_bboxes_per_image[:, 0] + 0.5 * gt_bboxes_per_image[:, 2])
        .unsqueeze(1)
        .repeat(1, total_num_anchors)
    )
    gt_bboxes_per_image_t = (
        (gt_bboxes_per_image[:, 1] - 0.5 * gt_bboxes_per_image[:, 3])
        .unsqueeze(1)
        .repeat(1, total_num_anchors)
    )
    gt_bboxes_per_image_b = (
        (gt_bboxes_per_image[:, 1] + 0.5 * gt_bboxes_per_image[:, 3])
        .unsqueeze(1)
        .repeat(1, total_num_anchors)
    )

    # 找出中心点在gt中的grid（许多）
    # grid中心点到gt四边的垂直距离
    b_l = x_centers_per_image - gt_bboxes_per_image_l
    b_r = gt_bboxes_per_image_r - x_centers_per_image
    b_t = y_centers_per_image - gt_bboxes_per_image_t
    b_b = gt_bboxes_per_image_b - y_centers_per_image
    bbox_deltas = torch.stack([b_l, b_t, b_r, b_b], 2)

    # is_in_boxes: grid中心点是否在特定gt中 [num_gt, total_num_anchors]
    # is_in_boxes_all: grid中心点是否在任一gt中 [total_num_anchors]
    is_in_boxes = bbox_deltas.min(dim=-1).values > 0.0
    is_in_boxes_all = is_in_boxes.sum(dim=0) > 0

    # in fixed center
    center_radius = 2.5

    # gt中心点与 center_radius * stride 垂直距离组成新的box筛选grid中心点
    gt_bboxes_per_image_l = (gt_bboxes_per_image[:, 0]).unsqueeze(1).repeat(
        1, total_num_anchors
    ) - center_radius * expanded_strides_per_image.unsqueeze(0)
    gt_bboxes_per_image_r = (gt_bboxes_per_image[:, 0]).unsqueeze(1).repeat(
        1, total_num_anchors
    ) + center_radius * expanded_strides_per_image.unsqueeze(0)
    gt_bboxes_per_image_t = (gt_bboxes_per_image[:, 1]).unsqueeze(1).repeat(
        1, total_num_anchors
    ) - center_radius * expanded_strides_per_image.unsqueeze(0)
    gt_bboxes_per_image_b = (gt_bboxes_per_image[:, 1]).unsqueeze(1).repeat(
        1, total_num_anchors
    ) + center_radius * expanded_strides_per_image.unsqueeze(0)

    c_l = x_centers_per_image - gt_bboxes_per_image_l
    c_r = gt_bboxes_per_image_r - x_centers_per_image
    c_t = y_centers_per_image - gt_bboxes_per_image_t
    c_b = gt_bboxes_per_image_b - y_centers_per_image
    center_deltas = torch.stack([c_l, c_t, c_r, c_b], 2)
    # 找出中心点在gt的中心点center_radius范围内的grid（少量）
    is_in_centers = center_deltas.min(dim=-1).values > 0.0
    is_in_centers_all = is_in_centers.sum(dim=0) > 0

    # in boxes and in centers
    is_in_boxes_or_center = is_in_boxes_all | is_in_centers_all

    is_in_boxes_and_center = (
            is_in_boxes[:, is_in_boxes_or_center] & is_in_centers[:, is_in_boxes_or_center]
    )
    return is_in_boxes_or_center, is_in_boxes_and_center


def dynamic_k_matching(fg_mask, cost, pair_wise_ious, gt_classes, num_gt):
    """
    :param fg_mask: 所有anchor中初步符合的anchor mask
    :param cost: anchors的损失矩阵
    :param pair_wise_ious: anchors与各个ground truth的iou
    :param gt_classes:
    :param num_gt:
    :return:
        fg_mask: 初步符合的anchor中最终符合的anchor mask
        num_fg: 最终参与预测的anchor的数量
        matched_gt_inds: 参与预测的anchor所对应的ground truth
        gt_matched_classes: 参与预测的anchor各自所属的类别（跟随ground truth）
        pred_ious_this_matching: 参与预测的anchor与其所对应的ground truth的iou
    """
    # ---------------------------------------------------------------
    matching_matrix = torch.zeros_like(cost)

    ious_in_boxes_matrix = pair_wise_ious
    n_candidate_k = min(10, ious_in_boxes_matrix.size(1))
    # topk_ious, _ = torch.topk(ious_in_boxes_matrix, n_candidate_k, dim=1)  # 得到每个gt的最大的前k个iou
    sorted_ious, indices = ious_in_boxes_matrix.sort(descending=True)
    topk_ious = sorted_ious[:, :n_candidate_k]
    dynamic_ks = torch.clamp(topk_ious.sum(1).int(), min=1)  # 将最大iou相加取整得到k
    for gt_idx in range(num_gt):
        _, pos_idx = cost[gt_idx].sort()
        if dynamic_ks[gt_idx].item() < len(pos_idx) - 1:
            pos_idx = pos_idx[:dynamic_ks[gt_idx].item()]
        # _, pos_idx = torch.topk(  # If largest is False then the smallest k elements are returned.
        #     cost[gt_idx], k=dynamic_ks[gt_idx].detach(), largest=False
        # )  # 返回k个最小cost的anchor的索引
        matching_matrix[gt_idx][pos_idx] = 1.0

    del topk_ious, dynamic_ks, pos_idx

    anchor_matching_gt = matching_matrix.sum(0)
    if (anchor_matching_gt > 1).sum() > 0:
        _, cost_argmin = torch.min(cost[:, anchor_matching_gt > 1], dim=0)
        matching_matrix[:, anchor_matching_gt > 1] *= 0.0
        matching_matrix[cost_argmin, anchor_matching_gt > 1] = 1.0
    fg_mask_inboxes = matching_matrix.sum(0) > 0.0
    num_fg = fg_mask_inboxes.sum().detach()

    # 最终符合的anchor mask
    fg_mask[fg_mask.clone()] = fg_mask_inboxes
    # 先mask选择有分配的anchor，再取索引得到所属gt
    matched_gt_inds = matching_matrix[:, fg_mask_inboxes].argmax(0)  # 每个gt的k个anchor分给gt认领类别
    # k个anchor的类别获得
    gt_matched_classes = gt_classes[matched_gt_inds]

    pred_ious_this_matching = (matching_matrix * pair_wise_ious).sum(0)[
        fg_mask_inboxes
    ]  # k个anchor的ious获得
    return fg_mask, num_fg, matched_gt_inds, gt_matched_classes, pred_ious_this_matching


def get_l1_type(l1_target, gt, stride, x_shifts, y_shifts, eps=1e-8):
    l1_target[:, 0] = gt[:, 0] / stride - x_shifts
    l1_target[:, 1] = gt[:, 1] / stride - y_shifts
    l1_target[:, 2] = torch.log(gt[:, 2] / stride + eps)
    l1_target[:, 3] = torch.log(gt[:, 3] / stride + eps)
    return l1_target
