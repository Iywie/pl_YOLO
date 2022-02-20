import torch


def MyEvaluator_step(detections, labels, hws, ids, val_size):
    nlabel = (labels.sum(dim=2) > 0).sum(dim=1)
    correct = 0
    num_gt_batch = nlabel.sum()
    img_h = hws[0]
    img_w = hws[1]
    for i in range(len(detections)):
        if detections[i] is None:
            continue
        detection = detections[i]
        label = labels[i]
        image_id = ids[i]

        bboxes = detection[:, 0:4]
        scale = min(
            val_size[0] / float(img_w[0]), val_size[1] / float(img_h[1])
        )
        bboxes /= scale

        num_gt_img = int(nlabel[i])
        gt_class = label[:num_gt_img, 0]
        gt_bboxes = label[:num_gt_img, 1:5]
        box_corner = gt_bboxes.new(gt_bboxes.shape)
        box_corner[:, 0] = gt_bboxes[:, 0] - gt_bboxes[:, 2] / 2
        box_corner[:, 1] = gt_bboxes[:, 1] - gt_bboxes[:, 3] / 2
        box_corner[:, 2] = gt_bboxes[:, 0] + gt_bboxes[:, 2] / 2
        box_corner[:, 3] = gt_bboxes[:, 1] + gt_bboxes[:, 3] / 2
        gt_bboxes = box_corner[:, :4]
        for gt_idx in range(num_gt_img):
            pred_boxes_gt = bboxes[detection[:, 6] == gt_class[gt_idx]]
            if pred_boxes_gt.shape[0] == 0:
                continue
            ious = bbox_iou(pred_boxes_gt, gt_bboxes[gt_idx].unsqueeze(0))
            for iou_idx in range(ious.shape[0]):
                if ious[iou_idx] >= 0.5:
                    correct += 1
                    break

    map_batch = float(correct / num_gt_batch)
    return map_batch, correct, num_gt_batch


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


