import io
import json
import torch
import numpy as np
import tempfile
import contextlib
# from pycocotools.cocoeval import COCOeval
from models.data.datasets.pycocotools.cocoeval import COCOeval


def COCOEvaluator(json_list, val_dataset):
    # detections: (x1, y1, x2, y2, obj_conf, class_conf, class)
    cocoGt = val_dataset.coco
    # pycocotools box format: (x1, y1, w, h)
    annType = ["segm", "bbox", "keypoints"]

    if len(json_list) > 0:
        _, tmp = tempfile.mkstemp()
        json.dump(json_list, open(tmp, "w"), skipkeys=True, ensure_ascii=True)
        cocoDt = cocoGt.loadRes(tmp)

        coco_pred = {"images": [], "categories": []}
        for (k, v) in cocoGt.imgs.items():
            coco_pred["images"].append(v)
        for (k, v) in cocoGt.cats.items():
            coco_pred["categories"].append(v)
        coco_pred["annotations"] = json_list
        # json.dump(coco_pred, open("./COCO_val.json", "w"))

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


def format_outputs(outputs, ids, hws, val_size, class_ids):
    """
    outputs: [batch, [x1, y1, x2, y2, confidence, class_pred]]
    """

    json_list = []
    data_list = [[np.empty(shape=[0, 5]) for _ in range(len(class_ids))] for _ in range(len(outputs))]
    for i, (output, img_h, img_w, img_id) in enumerate(zip(outputs, hws[0], hws[1], ids)):
        if output is None:
            data_list[i].append(img_id)
            continue

        bboxes = output[:, 0:4]
        # preprocessing: resize
        scale = min(val_size[0] / float(img_w), val_size[1] / float(img_h))
        bboxes /= scale
        coco_bboxes = xyxy2xywh(bboxes.clone())

        scores = output[:, 4]
        cls = output[:, 5]

        for ind in range(bboxes.shape[0]):
            label = class_ids[int(cls[ind])]
            pred_data = {
                "image_id": int(img_id),
                "category_id": label,
                "bbox": coco_bboxes[ind].cpu().numpy().tolist(),
                "score": scores[ind].cpu().numpy().item(),
                "segmentation": [],
            }  # COCO json format
            json_list.append(pred_data)

        for ind in range(bboxes.shape[0]):
            label = int(cls[ind])
            bbox = bboxes[ind].cpu().numpy()
            score = scores[ind].cpu().numpy()
            pred = np.append(bbox, score)
            pred = np.expand_dims(pred, axis=0)
            data_list[i][label] = np.append(data_list[i][label], pred, axis=0)
        data_list[i].append(img_id)

    return json_list, data_list


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
