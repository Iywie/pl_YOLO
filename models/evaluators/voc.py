import time
import numpy as np
from collections import Counter
from models.utils.bbox import box_iou


def evaluate(gt_boxes, det_boxes, iou_threshold=0.5):
    """
    gt_boxes: (list) {'img_id', 'box', 'cls'}
    det_boxes: (list) {'img_id', 'box', 'cls', 'score'}
    """

    # 1.Get classes of all bounding boxes 
    # 2.Separating them by classes
    gt_classes_only = []
    cls_bbs = {}
    for bb in gt_boxes:
        c = bb['cls']
        gt_classes_only.append(c)
        cls_bbs.setdefault(c, {'gts': [], 'dets': []})
        cls_bbs[c]['gts'].append(bb)
    gt_classes_only = list(set(gt_classes_only))
    for bb in det_boxes:
        c = bb['cls']
        cls_bbs.setdefault(c, {'gts': [], 'dets': []})
        cls_bbs[c]['dets'].append(bb)

    print('SE')

    # 1.For each class
    for c, v in cls_bbs.items():
        if c not in gt_classes_only:
            continue
        npos = len(v['gts'])
        # sort detections by decreasing confidence
        dets = [bb for bb in sorted(v['dets'], key=lambda b: b['score'], reverse=True)]
        # create dictionary with amount of expected detections for each image
        detected_gt_per_image = Counter([bb['img_id'] for bb in gt_boxes])
        for key, val in detected_gt_per_image.items():
            detected_gt_per_image[key] = np.zeros(val)
        print(f'Evaluating class: {c}')

        dict_table = {
            'img_id': [],
            'confidence': [],
            'TP': [],
            'FP': [],
            'acc TP': [],
            'acc FP': [],
            'precision': [],
            'recall': []
        }
        # Loop through detections
        TP = np.zeros(len(dets))
        FP = np.zeros(len(dets))
        for idx_det, det in enumerate(dets):
            img_id = det['img_id']
            dict_table['image'].append(img_id)
            dict_table['confidence'].append(det['score'])
            # Find ground truth image
            gt = [gt for gt in cls_bbs[c]['gt'] if gt['img_id'] == img_id]
            # Get the maximum iou among all detectins in the image
            iouMax = 0.0
            # Given the detection det, find ground-truth with the highest iou
            for j, g in enumerate(gt):
                # print('Ground truth gt => %s' str(g.get_absolute_bounding_box(format=BBFormat.XYX2Y2)))
                iou = box_iou(det, g)
                if iou > iouMax:
                    iouMax = iou
                    id_match_gt = j

            # Assign detection as TP or FP
            if iouMax >= iou_threshold:
                # gt was not matched with any detection
                if detected_gt_per_image[img_id][id_match_gt] == 0:
                    TP[idx_det] = 1  # detection is set as true positive
                    detected_gt_per_image[img_id][id_match_gt] = 1  # set flag to identify gt as already 'matched'
                    # print("TP")
                    dict_table['TP'].append(1)
                    dict_table['FP'].append(0)
                else:
                    FP[idx_det] = 1  # detection is set as false positive
                    dict_table['FP'].append(1)
                    dict_table['TP'].append(0)
                    # print("FP")
            # - A detected "cat" is overlaped with a GT "cat" with IOU >= iou_threshold.
            else:
                FP[idx_det] = 1  # detection is set as false positive
                dict_table['FP'].append(1)
                dict_table['TP'].append(0)
                # print("FP")