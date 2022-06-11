import io
import json
import torch
import numpy as np
import tempfile
import contextlib
# from pycocotools.cocoeval import COCOeval
from models.data.datasets.pycocotools.cocoeval import COCOeval


def COCOEvaluator(data_dict, val_dataset):
    # detections: (x1, y1, x2, y2, obj_conf, class_conf, class)
    cocoGt = val_dataset.coco
    # pycocotools box format: (x1, y1, w, h)
    annType = ["segm", "bbox", "keypoints"]

    if len(data_dict) > 0:
        json.dump(data_dict, open("./COCO_val.json", "w"))
        cocoDt = cocoGt.loadRes("./COCO_val.json")

        coco_pred = {"images": [], "categories": []}
        for (k, v) in cocoGt.imgs.items():
            coco_pred["images"].append(v)
        for (k, v) in cocoGt.cats.items():
            coco_pred["categories"].append(v)
        coco_pred["annotations"] = data_dict
        json.dump(coco_pred, open("./COCO_val.json", "w"))

        cocoEval = COCOeval(cocoGt, cocoDt, annType[1])
        cocoEval.evaluate()
        cocoEval.accumulate()
        redirect_string = io.StringIO()
        with contextlib.redirect_stdout(redirect_string):
            cocoEval.summarize()
        info = redirect_string.getvalue()
        return cocoEval.stats[0], cocoEval.stats[1], info
    else:
        return 0.0, 0.0, "No detection!"


def convert_to_coco_format(outputs, ids, hws, val_size, class_ids):
    data_list = []
    for (output, img_h, img_w, img_id) in zip(
        outputs, hws[0], hws[1], ids
    ):
        if output is None:
            continue

        bboxes = output[:, 0:4]

        # preprocessing: resize
        scale = min(
            val_size[0] / float(img_h), val_size[1] / float(img_w)
        )
        bboxes /= scale
        bboxes = xyxy2xywh(bboxes)
        # bboxes = xyxy2cxcywh(bboxes)
        # bboxes[:, :2] -= bboxes[:, 2:] / 2  # xy center to top-left corner

        cls = output[:, 6]
        scores = output[:, 4] * output[:, 5]
        # cls = output[:, 5]
        # scores = output[:, 4]
        for ind in range(bboxes.shape[0]):
            label = class_ids[int(cls[ind])]
            pred_data = {
                "image_id": int(img_id),
                "category_id": label,
                "bbox": bboxes[ind].cpu().numpy().tolist(),
                "score": scores[ind].cpu().numpy().item(),
                "segmentation": [],
            }  # COCO json format
            data_list.append(pred_data)
    return data_list


def xyxy2xywh(bboxes):
    bboxes[:, 2] = bboxes[:, 2] - bboxes[:, 0]
    bboxes[:, 3] = bboxes[:, 3] - bboxes[:, 1]
    return bboxes


def xyxy2cxcywh(x):
    # Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = (x[:, 0] + x[:, 2]) / 2  # x center
    y[:, 1] = (x[:, 1] + x[:, 3]) / 2  # y center
    y[:, 2] = x[:, 2] - x[:, 0]  # width
    y[:, 3] = x[:, 3] - x[:, 1]  # height
    return y
