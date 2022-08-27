import math
import torch
import torch.nn as nn


class YOLOv5Loss:
    def __init__(self, num_classes, img_size, anchors, strides, anchor_thre, balance):
        super(YOLOv5Loss, self).__init__()
        self.num_classes = num_classes
        self.img_size = img_size
        self.anchors = torch.tensor(anchors)
        self.strides = strides
        self.anchor_thre = anchor_thre
        self.balance = balance

        self.gr = 1.0
        self.BCEcls = None
        self.BCEobj = None
        self.cn = 0.0  # class negative
        self.pn = 1.0  # class positive
        self.lambda_box = 0.05
        self.lambda_obj = 1.0
        self.lambda_cls = 0.0375

    def __call__(self, inputs, targets):
        """
        :param inputs: a list of prediction grids
        :param targets: (bs, max_label, 5)
        :return:
        """
        if self.BCEobj is None:
            self.BCEcls = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([1.0]).type_as(targets))
            self.BCEobj = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([1.0]).type_as(targets))
        nl = len(inputs)    # number of layers
        na = len(self.anchors[0])   # number of anchors, every layer have the same number of anchors
        n_gt = (targets.sum(dim=2) > 0).sum(dim=1)
        nts = n_gt.sum()  # number of ground truths of the batch
        bs = targets.shape[0]
        lbox = torch.zeros(1).type_as(targets)
        lobj = torch.zeros(1).type_as(targets)
        lcls = torch.zeros(1).type_as(targets)

        for i in range(nl):
            prediction = inputs[i].view(bs, na, 5+self.num_classes, inputs[i].size(2), inputs[i].size(3)) \
                .permute(0, 1, 3, 4, 2).contiguous()
            inputs[i] = prediction

        targets = target2percent(targets, self.img_size)
        gts_list = []
        for img_idx in range(bs):
            nt = n_gt[img_idx]
            gt_boxes = targets[img_idx, :nt, 1:5]
            gt_classes = targets[img_idx, :nt, 0].unsqueeze(-1)
            gt_img_ids = torch.ones_like(gt_classes).type_as(gt_classes) * img_idx
            gt = torch.cat((gt_img_ids, gt_classes, gt_boxes), 1)
            gts_list.append(gt)
        targets = torch.cat(gts_list, 0)

        ai = torch.arange(na).type_as(targets)  # (na) anchor indice
        ai = ai.view(na, 1).repeat(1, nts)  # (na,1)

        # add the anchor indices in the last of prediction (6 to 7)
        targets = targets.repeat(na, 1, 1)  # (nt,6) to (na,nt,6)
        targets = torch.cat((targets, ai[:, :, None]), 2)  # (na,nt,7)

        g = 0.5  # bias
        off = torch.tensor([[0, 0], [1, 0], [0, 1], [-1, 0], [0, -1]]).type_as(targets)  # (5,2)
        off = off * g

        gain = torch.ones(7).type_as(targets)
        tcls, tbox, indices, anch = [], [], [], []
        for i in range(nl):
            # Both anchor and target move from image to feature map
            anchor = self.anchors[i].type_as(targets)
            # Anchor is a scale
            anchor = anchor / self.strides[i]
            # YOLO format is percentage. Turn it to absolute position in different feature maps.
            gain[2:6] = torch.tensor(inputs[i].shape)[[3, 2, 3, 2]]  # xyxy gain
            t = targets * gain

            if nts:
                # choose targets which don't have large gap (factor)
                r = t[:, :, 4:6] / anchor[:, None]  # wh ratio
                j = torch.max(r, 1. / r).max(2)[0] < self.anchor_thre

                t = t[j]

                # Center point position on feature map
                gxy = t[:, 2:4]  # grid xy
                gxi = gain[[2, 3]] - gxy  # Inverse

                j, k = ((gxy % 1. < g) & (gxy > 1.)).T
                l, m = ((gxi % 1. < g) & (gxi > 1.)).T
                j = torch.stack((torch.ones_like(j), j, k, l, m))
                t = t.repeat((5, 1, 1))[j]
                offsets = (torch.zeros_like(gxy)[None] + off[:, None])[j]

            else:
                t = targets[0]
                offsets = 0
            # Define
            b, c = t[:, :2].long().T  # image, class
            gxy = t[:, 2:4]  # grid xy
            gwh = t[:, 4:6]  # grid wh
            gij = (gxy - offsets).long()
            gi, gj = gij.T  # grid xy indices

            # Append
            a = t[:, 6].long()  # anchor indices
            indices.append(
                (b, a, gj.clamp_(0, gain[3] - 1), gi.clamp_(0, gain[2] - 1)))  # image, anchor, grid indices
            tbox.append(torch.cat((gxy - gij, gwh), 1))  # box
            anch.append(anchor[a])  # anchors
            tcls.append(c)  # class

        for i, pi in enumerate(inputs):  # layer index, layer predictions
            b, a, gj, gi = indices[i]  # image, anchor, gridy, gridx
            tobj = torch.zeros_like(pi[..., 0]).type_as(pi)
            n = b.shape[0]  # number of targets
            if n:
                ps = pi[b, a, gj, gi]  # prediction subset corresponding to targets
                # Regression
                pxy = ps[:, :2].sigmoid() * 2. - 0.5
                pwh = (ps[:, 2:4].sigmoid() * 2) ** 2 * anch[i]
                pbox = torch.cat((pxy, pwh), 1)  # predicted box
                iou = bbox_iou(pbox.T, tbox[i], x1y1x2y2=False, CIoU=True)  # iou(prediction, target)
                lbox += (1.0 - iou).mean()  # iou loss

                # Objectness
                tobj[b, a, gj, gi] = (1.0 - self.gr) + self.gr * iou.detach().clamp(0).type_as(tobj)  # iou ratio

                # Classification
                if self.num_classes > 1:  # cls loss (only if multiple classes)
                    t = torch.full_like(ps[:, 5:], self.cn).type_as(pi)  # targets
                    t[range(n), tcls[i]] = self.pn
                    lcls += self.BCEcls(ps[:, 5:], t)  # BCE

            obji = self.BCEobj(pi[..., 4], tobj)
            lobj += obji * self.balance[i]  # obj loss
        lbox *= self.lambda_box
        lobj *= self.lambda_obj
        lcls *= self.lambda_cls
        loss = lbox + lobj + lcls
        return loss * bs, torch.cat((lbox, lobj, lcls, loss)).detach()


def target2percent(targets, img_size):
    # From [cx, cy, w, h] to [cx%, cy%, w%, h%]
    targets[:, :, 1] = targets[:, :, 1] / img_size[0]
    targets[:, :, 3] = targets[:, :, 3] / img_size[0]
    targets[:, :, 2] = targets[:, :, 2] / img_size[1]
    targets[:, :, 4] = targets[:, :, 4] / img_size[1]
    return targets


def bbox_iou(box1, box2, x1y1x2y2=True, GIoU=False, DIoU=False, CIoU=False, eps=1e-7):
    # Returns the IoU of box1 to box2. box1 is 4, box2 is nx4
    box2 = box2.T

    # Get the coordinates of bounding boxes
    if x1y1x2y2:  # x1, y1, x2, y2 = box1
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[2], box1[3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[1], box2[2], box2[3]
    else:  # transform from xywh to xyxy
        b1_x1, b1_x2 = box1[0] - box1[2] / 2, box1[0] + box1[2] / 2
        b1_y1, b1_y2 = box1[1] - box1[3] / 2, box1[1] + box1[3] / 2
        b2_x1, b2_x2 = box2[0] - box2[2] / 2, box2[0] + box2[2] / 2
        b2_y1, b2_y2 = box2[1] - box2[3] / 2, box2[1] + box2[3] / 2

    # Intersection area
    inter = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) * \
            (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(0)

    # Union Area
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
