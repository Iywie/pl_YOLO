import torch
import torchvision
from models.evaluators.nms import non_max_suppression


def coco_post(prediction, num_classes, conf_thre=0.7, nms_thre=0.45, class_agnostic=False):
    """
    The prediction is [center_x, center_y, h, w]
    Turn it to [x1, y1, x2, y2] because the NMS function need this.
    :param prediction: [batch, num_prediction, prediction]
        predicted box format: (center x, center y, w, h)
    :param num_classes:
    :param conf_thre:
    :param nms_thre:
    :param class_agnostic:
    :return: (x1, y1, x2, y2, obj_conf, class_conf, class)
    """
    if isinstance(prediction, list):
        prediction = torch.cat(prediction, 1)

    box_corner = prediction.new(prediction.shape)
    box_corner[:, :, 0] = prediction[:, :, 0] - prediction[:, :, 2] / 2
    box_corner[:, :, 1] = prediction[:, :, 1] - prediction[:, :, 3] / 2
    box_corner[:, :, 2] = prediction[:, :, 0] + prediction[:, :, 2] / 2
    box_corner[:, :, 3] = prediction[:, :, 1] + prediction[:, :, 3] / 2
    prediction[:, :, :4] = box_corner[:, :, :4]

    output = [None for _ in range(len(prediction))]
    for i, image_pred in enumerate(prediction):

        # If none are remaining => process next image
        if not image_pred.size(0):
            continue
        # Get score and class with highest confidence
        class_conf, class_pred = torch.max(image_pred[:, 5: 5 + num_classes], 1, keepdim=True)
        # Confidence = object possibility multiply class score
        # conf_mask = (image_pred[:, 4] * class_conf.squeeze() >= conf_thre).squeeze()
        conf_mask = (image_pred[:, 4] * class_conf.squeeze() >= conf_thre).squeeze()
        # Detections ordered as (x1, y1, x2, y2, obj_conf, class_conf, class)
        detections = torch.cat((image_pred[:, :5], class_conf, class_pred.float()), 1)
        detections = detections[conf_mask]
        if not detections.size(0):
            continue

        # NMS
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
        # detections = non_max_suppression
        detections = detections[nms_out_index]
        output[i] = detections

    return output
