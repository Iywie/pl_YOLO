import os
import io
import json
import tempfile
import contextlib
import torch
import torchvision
from pycocotools.coco import COCO


def postprocess(prediction, num_classes, conf_thre=0.7, nms_thre=0.45, class_agnostic=False):
    """
    The prediction is [center_x, center_y, h, w]
    First turn it to [x1, y1, x2, y2] because the NMS function need this.
    :param prediction: [batch, num_prediction, prediction]
        predicted box format: (center x, center y, w, h)
    :param num_classes:
    :param conf_thre:
    :param nms_thre:
    :param class_agnostic:
    :return: (x1, y1, x2, y2, obj_conf, class_conf, class)
    """
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

        detections = detections[nms_out_index]
        output[i] = detections

    return output


def COCOEvaluator(
    detection_list,
    images_id,
    images_hw,
    img_size_val,
    data_dir,
    json_file,
):

    # detections: (x1, y1, x2, y2, obj_conf, class_conf, class)
    data_list = []
    cocoGt = COCO(os.path.join(data_dir, "annotations", json_file))
    class_ids = sorted(cocoGt.getCatIds())
    for i in range(len(detection_list)):
        data_list.extend(convert_to_coco_format(detection_list[i], images_id[i], images_hw[i], img_size_val, class_ids))
        # coco box format: (x1, y1, w, h)
    eval_results = evaluate_prediction(data_list, cocoGt)
    return eval_results


def evaluate_prediction(data_dict, cocoGt):
    annType = ["segm", "bbox", "keypoints"]
    if len(data_dict) > 0:
        _, tmp = tempfile.mkstemp()
        json.dump(data_dict, open(tmp, "w"))
        cocoDt = cocoGt.loadRes(tmp)
        from pycocotools.cocoeval import COCOeval
        cocoEval = COCOeval(cocoGt, cocoDt, annType[1])
        cocoEval.evaluate()
        cocoEval.accumulate()
        redirect_string = io.StringIO()
        with contextlib.redirect_stdout(redirect_string):
            cocoEval.summarize()
        info = redirect_string.getvalue()
        return cocoEval.stats[0], cocoEval.stats[1], info
    else:
        return 0, 0, None


def convert_to_coco_format(outputs, ids, hws, val_size, class_ids):
    data_list = []
    for (output, img_h, img_w, img_id) in zip(
        outputs, hws[0], hws[1], ids
    ):
        if output is None:
            continue
        output = output.cpu()

        bboxes = output[:, 0:4]

        # preprocessing: resize
        scale = min(
            val_size[0] / float(img_h), val_size[1] / float(img_w)
        )
        bboxes /= scale
        bboxes = xyxy2xywh(bboxes)

        cls = output[:, 6]
        scores = output[:, 4] * output[:, 5]
        for ind in range(bboxes.shape[0]):
            label = class_ids[int(cls[ind])]
            pred_data = {
                "image_id": int(img_id),
                "category_id": label,
                "bbox": bboxes[ind].numpy().tolist(),
                "score": scores[ind].numpy().item(),
                "segmentation": [],
            }  # COCO json format
            data_list.append(pred_data)
    return data_list


def xyxy2xywh(bboxes):
    bboxes[:, 2] = bboxes[:, 2] - bboxes[:, 0]
    bboxes[:, 3] = bboxes[:, 3] - bboxes[:, 1]
    return bboxes
