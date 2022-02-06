import os
import io
import json
import tempfile
import contextlib
from pycocotools.cocoeval import COCOeval


def COCOEvaluator(
    detection_list,
    images_id,
    images_hw,
    img_size_val,
    val_dataset
):

    # detections: (x1, y1, x2, y2, obj_conf, class_conf, class)
    data_list = []
    cocoGt = val_dataset.coco
    for i in range(len(detection_list)):
        data_list.extend(
            convert_to_coco_format(detection_list[i], images_id[i], images_hw[i], img_size_val, val_dataset.ids)
        )
        # coco box format: (x1, y1, w, h)
    eval_results = evaluate_prediction(data_list, cocoGt)
    return eval_results


def evaluate_prediction(data_dict, cocoGt):
    annType = ["segm", "bbox", "keypoints"]
    if len(data_dict) > 0:
        _, tmp = tempfile.mkstemp()
        json.dump(data_dict, open(tmp, "w"))
        cocoDt = cocoGt.loadRes(tmp)

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