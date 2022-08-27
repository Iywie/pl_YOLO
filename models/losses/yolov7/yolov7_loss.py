import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.layers.losses.iou_loss import bboxes_iou
from models.utils.bbox import xywh2xyxy


class YOLOv7Loss(nn.Module):
    def __init__(self,
                 num_classes,
                 strides,
                 anchors,
                 label_smoothing=0,
                 focal_g=0.0,):
        super(YOLOv7Loss, self).__init__()

        self.anchors = torch.tensor(anchors)
        self.num_classes = num_classes
        self.strides = strides

        self.nl = len(strides)
        self.na = len(anchors[0])
        self.ch = 5 + self.num_classes

        self.balance = [0.4, 1.0, 4]
        self.box_ratio = 0.05
        self.obj_ratio = 1
        self.cls_ratio = 0.5 * (num_classes / 80)
        self.threshold = 4.0

        self.grids = [torch.zeros(1)] * len(strides)
        self.anchor_grid = self.anchors.clone().view(self.nl, 1, -1, 1, 1, 2)

        self.cp, self.cn = smooth_BCE(eps=label_smoothing)
        self.BCEcls, self.BCEobj, self.gr = nn.BCEWithLogitsLoss(), nn.BCEWithLogitsLoss(), 1

    def __call__(self, inputs, targets):
        # input of inputs: [batch, ch * anchor, h, w]

        batch_size = targets.shape[0]
        # input: [batch, anchor, h, w, ch]
        for i in range(self.nl):
            prediction = inputs[i].view(
                inputs[i].size(0), self.na, self.ch, inputs[i].size(2), inputs[i].size(3)
            ).permute(0, 1, 3, 4, 2).contiguous()
            inputs[i] = prediction

        # inference
        if not self.training:
            preds = []
            for i in range(self.nl):
                pred = inputs[i].sigmoid()
                h, w = pred.shape[2:4]
                # Three steps to localize predictions: grid, shifts of x and y, grid with stride
                if self.grids[i].shape[2:4] != pred.shape[2:4]:
                    yv, xv = torch.meshgrid([torch.arange(h), torch.arange(w)], indexing='ij')
                    grid = torch.stack((xv, yv), 2).view(1, 1, h, w, 2).type_as(pred)
                    self.grids[i] = grid
                else:
                    grid = self.grids[i]

                pred[..., :2] = (pred[..., :2] * 2. - 0.5 + grid) * self.strides[i]
                pred[..., 2:4] = (pred[..., 2:4] * 2) ** 2 * self.anchor_grid[i].type_as(pred)
                pred = pred.reshape(batch_size, -1, self.ch)
                preds.append(pred)

            # preds: [batch_size, all predictions, n_ch]
            predictions = torch.cat(preds, 1)
            # from (cx,cy,w,h) to (x1,y1,x2,y2)
            box_corner = predictions.new(predictions.shape)
            box_corner = box_corner[:, :, 0:4]
            box_corner[:, :, 0] = predictions[:, :, 0] - predictions[:, :, 2] / 2
            box_corner[:, :, 1] = predictions[:, :, 1] - predictions[:, :, 3] / 2
            box_corner[:, :, 2] = predictions[:, :, 0] + predictions[:, :, 2] / 2
            box_corner[:, :, 3] = predictions[:, :, 1] + predictions[:, :, 3] / 2
            predictions[:, :, :4] = box_corner[:, :, :4]
            return predictions

        # Compute loss
        # Processing ground truth to tensor (img_idx, class, cx, cy, w, h)
        n_gt = (targets.sum(dim=2) > 0).sum(dim=1)

        gts_list = []
        for img_idx in range(batch_size):
            nt = n_gt[img_idx]
            gt_boxes = targets[img_idx, :nt, 1:5]
            gt_classes = targets[img_idx, :nt, 0].unsqueeze(-1)
            gt_img_ids = torch.ones_like(gt_classes).type_as(gt_classes) * img_idx
            gt = torch.cat((gt_img_ids, gt_classes, gt_boxes), 1)
            gts_list.append(gt)
        targets = torch.cat(gts_list, 0)

        bs, as_, gjs, gis, targets, anchors = self.build_targets(inputs, targets)

        cls_loss = torch.zeros(1).type_as(inputs[0])
        box_loss = torch.zeros(1).type_as(inputs[0])
        obj_loss = torch.zeros(1).type_as(inputs[0])

        for i, prediction in enumerate(inputs):
            #   image, anchor, gridy, gridx
            b, a, gj, gi = bs[i], as_[i], gjs[i], gis[i]
            tobj = torch.zeros_like(prediction[..., 0]).type_as(prediction)  # target obj

            n = b.shape[0]
            if n:
                prediction_pos = prediction[b, a, gj, gi]  # prediction subset corresponding to targets

                grid = torch.stack([gi, gj], dim=1)

                #   进行解码，获得预测结果
                xy = prediction_pos[:, :2].sigmoid() * 2. - 0.5
                wh = (prediction_pos[:, 2:4].sigmoid() * 2) ** 2 * anchors[i]
                box = torch.cat((xy, wh), 1)

                #   对真实框进行处理，映射到特征层上
                selected_tbox = targets[i][:, 2:6] / self.strides[i]
                selected_tbox[:, :2] = selected_tbox[:, :2] - grid.type_as(prediction)

                #   计算预测框和真实框的回归损失
                iou = bbox_iou(box.T, selected_tbox, x1y1x2y2=False, CIoU=True)
                box_loss += (1.0 - iou).mean()
                # -------------------------------------------#
                #   根据预测结果的iou获得置信度损失的gt
                # -------------------------------------------#
                tobj[b, a, gj, gi] = (1.0 - self.gr) + self.gr * iou.detach().clamp(0).type(tobj.dtype)  # iou ratio

                # -------------------------------------------#
                #   计算匹配上的正样本的分类损失
                # -------------------------------------------#
                selected_tcls = targets[i][:, 1].long()
                t = torch.full_like(prediction_pos[:, 5:], self.cn).type_as(prediction)  # targets
                t[range(n), selected_tcls] = self.cp
                cls_loss += self.BCEcls(prediction_pos[:, 5:], t)  # BCE

            # -------------------------------------------#
            #   计算目标是否存在的置信度损失
            #   并且乘上每个特征层的比例
            # -------------------------------------------#
            obj_loss += self.BCEobj(prediction[..., 4], tobj) * self.balance[i]  # obj loss

        # -------------------------------------------#
        #   将各个部分的损失乘上比例
        #   全加起来后，乘上batch_size
        # -------------------------------------------#
        box_loss *= self.box_ratio
        obj_loss *= self.obj_ratio
        cls_loss *= self.cls_ratio

        loss = box_loss + obj_loss + cls_loss

        losses = {"loss": loss}
        return losses

    def build_targets(self, predictions, targets):

        # indice: [img_idx, anchor_idx, grid_x, grid_y]
        indices, anch = self.find_3_positive(predictions, targets)

        matching_bs = [[] for _ in predictions]
        matching_as = [[] for _ in predictions]
        matching_gjs = [[] for _ in predictions]
        matching_gis = [[] for _ in predictions]
        matching_targets = [[] for _ in predictions]
        matching_anchs = [[] for _ in predictions]

        # label assignment for each image
        for batch_idx in range(predictions[0].shape[0]):

            # targets of this image
            b_idx = targets[:, 0] == batch_idx
            this_target = targets[b_idx]
            if this_target.shape[0] == 0:
                continue

            txywh = this_target[:, 2:6]
            txyxy = xywh2xyxy(txywh)

            pxyxys = []
            p_cls = []
            p_obj = []
            from_which_layer = []
            all_b = []
            all_a = []
            all_gj = []
            all_gi = []
            all_anch = []

            for i, map in enumerate(predictions):
                b, a, gj, gi = indices[i]
                idx = (b == batch_idx)
                b, a, gj, gi = b[idx], a[idx], gj[idx], gi[idx]
                all_b.append(b)
                all_a.append(a)
                all_gj.append(gj)
                all_gi.append(gi)
                all_anch.append(anch[i][idx])
                from_which_layer.append(torch.ones(size=(len(b),)) * i)

                fg_pred = map[b, a, gj, gi]
                p_obj.append(fg_pred[:, 4:5])
                p_cls.append(fg_pred[:, 5:])

                grid = torch.stack([gi, gj], dim=1)
                pxy = (fg_pred[:, :2].sigmoid() * 2. - 0.5 + grid) * self.strides[i]  # / 8.
                pwh = (fg_pred[:, 2:4].sigmoid() * 2) ** 2 * anch[i][idx] * self.strides[i]  # / 8.
                pxywh = torch.cat([pxy, pwh], dim=-1)
                pxyxy = xywh2xyxy(pxywh)
                pxyxys.append(pxyxy)

            pxyxys = torch.cat(pxyxys, dim=0)
            if pxyxys.shape[0] == 0:
                continue
            p_obj = torch.cat(p_obj, dim=0)
            p_cls = torch.cat(p_cls, dim=0)
            from_which_layer = torch.cat(from_which_layer, dim=0)
            all_b = torch.cat(all_b, dim=0)
            all_a = torch.cat(all_a, dim=0)
            all_gj = torch.cat(all_gj, dim=0)
            all_gi = torch.cat(all_gi, dim=0)
            all_anch = torch.cat(all_anch, dim=0)

            # Cost matrix
            pair_wise_iou = bboxes_iou(txyxy, pxyxys)
            pair_wise_iou_loss = -torch.log(pair_wise_iou + 1e-8)

            top_k, _ = torch.topk(pair_wise_iou, min(10, pair_wise_iou.shape[1]), dim=1)
            dynamic_ks = torch.clamp(top_k.sum(1).int(), min=1)

            gt_cls_per_image = (
                F.one_hot(this_target[:, 1].to(torch.int64), self.num_classes)
                .float()
                .unsqueeze(1)
                .repeat(1, pxyxys.shape[0], 1)
            )

            num_gt = this_target.shape[0]
            cls_preds_ = (
                    p_cls.float().unsqueeze(0).repeat(num_gt, 1, 1).sigmoid_()
                    * p_obj.unsqueeze(0).repeat(num_gt, 1, 1).sigmoid_()
            )

            y = cls_preds_.sqrt_()
            pair_wise_cls_loss = F.binary_cross_entropy_with_logits(
                torch.log(y / (1 - y)), gt_cls_per_image, reduction="none"
            ).sum(-1)
            del cls_preds_

            cost = (
                    pair_wise_cls_loss
                    + 3.0 * pair_wise_iou_loss
            )

            # Dynamic k
            matching_matrix = torch.zeros_like(cost)

            for gt_idx in range(num_gt):
                _, pos_idx = torch.topk(
                    cost[gt_idx], k=dynamic_ks[gt_idx].item(), largest=False
                )
                matching_matrix[gt_idx][pos_idx] = 1.0

            del top_k, dynamic_ks
            anchor_matching_gt = matching_matrix.sum(0)
            if (anchor_matching_gt > 1).sum() > 0:
                _, cost_argmin = torch.min(cost[:, anchor_matching_gt > 1], dim=0)
                matching_matrix[:, anchor_matching_gt > 1] *= 0.0
                matching_matrix[cost_argmin, anchor_matching_gt > 1] = 1.0
            fg_mask_inboxes = matching_matrix.sum(0) > 0.0
            matched_gt_inds = matching_matrix[:, fg_mask_inboxes].argmax(0)

            from_which_layer = from_which_layer[fg_mask_inboxes]
            all_b = all_b[fg_mask_inboxes]
            all_a = all_a[fg_mask_inboxes]
            all_gj = all_gj[fg_mask_inboxes]
            all_gi = all_gi[fg_mask_inboxes]
            all_anch = all_anch[fg_mask_inboxes]

            this_target = this_target[matched_gt_inds]

            for i in range(self.nl):
                layer_idx = from_which_layer == i
                matching_bs[i].append(all_b[layer_idx])
                matching_as[i].append(all_a[layer_idx])
                matching_gjs[i].append(all_gj[layer_idx])
                matching_gis[i].append(all_gi[layer_idx])
                matching_targets[i].append(this_target[layer_idx])
                matching_anchs[i].append(all_anch[layer_idx])

        for i in range(self.nl):
            if matching_targets[i]:
                matching_bs[i] = torch.cat(matching_bs[i], dim=0)
                matching_as[i] = torch.cat(matching_as[i], dim=0)
                matching_gjs[i] = torch.cat(matching_gjs[i], dim=0)
                matching_gis[i] = torch.cat(matching_gis[i], dim=0)
                matching_targets[i] = torch.cat(matching_targets[i], dim=0)
                matching_anchs[i] = torch.cat(matching_anchs[i], dim=0)
            else:
                matching_bs[i] = torch.tensor([]).type_as(targets)
                matching_as[i] = torch.tensor([]).type_as(targets)
                matching_gjs[i] = torch.tensor([]).type_as(targets)
                matching_gis[i] = torch.tensor([]).type_as(targets)
                matching_targets[i] = torch.tensor([]).type_as(targets)
                matching_anchs[i] = torch.tensor([]).type_as(targets)

        return matching_bs, matching_as, matching_gjs, matching_gis, matching_targets, matching_anchs

    def find_3_positive(self, predictions, targets):
        """
        Args:
            predictions(tensor): [nb, na, w, h, ch]
            targets(tensor): [image_idx, class, x, y, w, h]
        Return:
            indice: [img_idx, anchor_idx, grid_x, grid_y]
            anchor: [anchor_w, anchor_h]
        """
        na, nt = self.na, targets.shape[0]  # number of anchors, targets
        indices, anch = [], []
        gain = torch.ones(7).type_as(targets).long()  # normalized to gridspace gain
        ai = torch.arange(na).type_as(targets).view(na, 1).repeat(1, nt)  # same as .repeat_interleave(nt)
        targets = torch.cat((targets.repeat(na, 1, 1), ai[:, :, None]), 2)  # append anchor indices

        g = 0.5  # bias
        off = torch.tensor([[0, 0],
                            [1, 0], [0, 1], [-1, 0], [0, -1],  # j,k,l,m
                            # [1, 1], [1, -1], [-1, 1], [-1, -1],  # jk,jm,lk,lm
                            ], device=targets.device).float() * g  # offsets

        for i in range(self.nl):

            # put anchor and target to feature map
            anchors = (self.anchors[i] / self.strides[i]).type_as(predictions[i])
            gain[2:6] = self.strides[i]
            target = targets / gain
            gain[2:6] = torch.tensor(predictions[i].shape)[[3, 2, 3, 2]]  # w and h

            # Match targets to anchors
            if nt:
                # target and anchor wh ratio in threshold
                r = target[:, :, 4:6] / anchors[:, None]  # wh ratio
                wh_mask = torch.max(r, 1. / r).max(2)[0] < self.threshold  # compare
                t = target[wh_mask]

                # Positive adjacent grid
                gxy = t[:, 2:4]  # grid xy
                gxi = gain[[2, 3]] - gxy  # inverse grid xy
                j, k = ((gxy % 1. < g) & (gxy > 1.)).T
                l, m = ((gxi % 1. < g) & (gxi > 1.)).T
                j = torch.stack((torch.ones_like(j), j, k, l, m))
                t = t.repeat((5, 1, 1))[j]
                offsets = (torch.zeros_like(gxy)[None] + off[:, None])[j]
            else:
                t = targets[0]
                offsets = 0

            # Define
            b, c = t[:, :2].long().T  # image_idx, class
            gxy = t[:, 2:4]  # grid xy
            gwh = t[:, 4:6]  # grid wh
            gij = (gxy - offsets).long()
            gi, gj = gij.T  # grid xy indices

            # Append
            a = t[:, 6].long()  # anchor indices
            indices.append((b, a, gj.clamp_(0, gain[3] - 1), gi.clamp_(0, gain[2] - 1)))  # image, anchor, grid indices
            anch.append(anchors[a])  # anchors

        return indices, anch


def smooth_BCE(eps=0.1):  # https://github.com/ultralytics/yolov3/issues/238#issuecomment-598028441
    # return positive, negative label smoothing BCE targets
    return 1.0 - 0.5 * eps, 0.5 * eps


def bbox_iou(box1, box2, x1y1x2y2=True, GIoU=False, DIoU=False, CIoU=False, eps=1e-7):
    box2 = box2.T

    if x1y1x2y2:
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[2], box1[3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[1], box2[2], box2[3]
    else:
        b1_x1, b1_x2 = box1[0] - box1[2] / 2, box1[0] + box1[2] / 2
        b1_y1, b1_y2 = box1[1] - box1[3] / 2, box1[1] + box1[3] / 2
        b2_x1, b2_x2 = box2[0] - box2[2] / 2, box2[0] + box2[2] / 2
        b2_y1, b2_y2 = box2[1] - box2[3] / 2, box2[1] + box2[3] / 2

    inter = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) * \
            (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(0)

    w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
    w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps
    union = w1 * h1 + w2 * h2 - inter + eps

    iou = inter / union

    if GIoU or DIoU or CIoU:
        cw = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)  # convex (smallest enclosing box) width
        ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)  # convex height
        if CIoU or DIoU:  # Distance or Complete IoU https://arxiv.org/abs/1911.08287v1
            c2 = cw ** 2 + ch ** 2 + eps  # convex diagonal squared
            rho2 = ((b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2 +
                    (b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2) / 4  # center distance squared
            if DIoU:
                return iou - rho2 / c2  # DIoU
            elif CIoU:  # https://github.com/Zzh-tju/DIoU-SSD-pytorch/blob/master/utils/box/box_utils.py#L47
                v = (4 / math.pi ** 2) * torch.pow(torch.atan(w2 / h2) - torch.atan(w1 / h1), 2)
                with torch.no_grad():
                    alpha = v / (v - iou + (1 + eps))
                return iou - (rho2 / c2 + v * alpha)  # CIoU
        else:  # GIoU https://arxiv.org/pdf/1902.09630.pdf
            c_area = cw * ch + eps  # convex area
            return iou - (c_area - union) / c_area  # GIoU
    else:
        return iou  # IoU
