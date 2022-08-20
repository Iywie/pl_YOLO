import torch
import torchvision
import numpy as np
from models.utils.bbox import xyxy2xywh


def postprocess(predictions, conf_thre=0.7, nms_thre=0.45, class_agnostic=False):
    max_det = 300  # maximum number of detections per image
    max_nms = 10000  # maximum number of boxes into torchvision.ops.nms()

    output = [None for _ in range(predictions.shape[0])]
    for i in range(predictions.shape[0]):
        image_pred = predictions[i]
        # If none are remaining => process next image
        if not image_pred.shape[0]:
            continue
        # Get class and correspond score
        class_conf, class_pred = torch.max(image_pred[:, 5:], 1, keepdim=True)
        confidence = image_pred[:, 4] * class_conf.squeeze()
        conf_mask = (confidence >= conf_thre).squeeze()
        # Detections ordered as (x1, y1, x2, y2, confidence, class_pred)
        detections = torch.cat((image_pred[:, :4], confidence.unsqueeze(-1), class_pred.float()), 1)
        detections = detections[conf_mask]
        if detections.shape[0] > max_nms:
            detections = detections[:max_nms]
        if not detections.size(0):
            continue

        if class_agnostic:
            nms_out_index = torchvision.ops.nms(
                detections[:, :4],
                detections[:, 4],
                nms_thre,
            )
        else:
            nms_out_index = torchvision.ops.batched_nms(
                detections[:, :4],
                detections[:, 4],
                detections[:, 5],
                nms_thre,
            )

        detections = detections[nms_out_index]
        if detections.shape[0] > max_det:  # limit detections
            detections = detections[:max_det]
        output[i] = detections

    return output


def demo_postprocess(predictions, conf_thre=0.7, nms_thre=0.45, class_agnostic=False):
    max_det = 300  # maximum number of detections per image
    max_nms = 10000  # maximum number of boxes into torchvision.ops.nms()

    output = [None for _ in range(predictions.shape[0])]
    for i in range(predictions.shape[0]):
        image_pred = predictions[i]
        # If none are remaining => process next image
        if not image_pred.shape[0]:
            continue
        # Get class and correspond score
        class_conf, class_pred = torch.max(image_pred[:, 5:], 1, keepdim=True)
        confidence = image_pred[:, 4] * class_conf.squeeze()
        conf_mask = (confidence >= conf_thre).squeeze()
        # Detections ordered as (x1, y1, x2, y2, confidence, class_pred)
        detections = torch.cat((image_pred[:, :4], confidence.unsqueeze(-1), class_pred.float()), 1)
        detections = detections[conf_mask]
        if detections.shape[0] > max_nms:
            detections = detections[:max_nms]
        if not detections.size(0):
            continue

        if class_agnostic:
            nms_out_index = torchvision.ops.nms(
                detections[:, :4],
                detections[:, 4],
                nms_thre,
            )
        else:
            nms_out_index = torchvision.ops.batched_nms(
                detections[:, :4],
                detections[:, 4],
                detections[:, 5],
                nms_thre,
            )

        detections = detections[nms_out_index]
        if detections.shape[0] > max_det:  # limit detections
            detections = detections[:max_det]
        output[i] = detections

    return output


def format_outputs(outputs, ids, hws, val_size, class_ids, labels):
    """
    outputs: [batch, [x1, y1, x2, y2, confidence, class_pred]]
    """

    json_list = []
    # det_list (list[list]): shape(num_images, num_classes)
    det_list = [[np.empty(shape=[0, 5]) for _ in range(len(class_ids))] for _ in range(len(outputs))]

    # for each image
    for i, (output, img_h, img_w, img_id) in enumerate(zip(outputs, hws[0], hws[1], ids)):
        if output is None:
            continue

        bboxes = output[:, 0:4]
        scale = min(val_size[0] / float(img_w), val_size[1] / float(img_h))
        bboxes /= scale
        coco_bboxes = xyxy2xywh(bboxes)

        scores = output[:, 4]
        clses = output[:, 5]

        # COCO format follows the prediction
        for bbox, cocobox, score, cls in zip(bboxes, coco_bboxes, scores, clses):
            # COCO format
            cls = int(cls)
            class_id = class_ids[cls]
            pred_data = {
                "image_id": int(img_id),
                "category_id": class_id,
                "bbox": cocobox.cpu().numpy().tolist(),
                "score": score.cpu().numpy().item(),
                "segmentation": [],
            }
            json_list.append(pred_data)

        # VOC format follows the class
        for c in range(len(class_ids)):
            # detection np.array(x1, y1, x2, y2, score)
            det_ind = clses == c
            detections = output[det_ind, 0:5]
            det_list[i][c] = detections.cpu().numpy()

    return json_list, det_list
