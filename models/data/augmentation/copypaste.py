import random
import cv2
import numpy as np
from models.utils.bbox import bbox_ioa


def copyPaste(img, labels, objects, copypaste_scale):

    if len(labels):
        clss = labels[:, 4]
        clss = np.unique(clss)
        clss = clss.astype(int)
    else:
        return img.astype(np.uint8), labels

    img_h, img_w = img.shape[:2]
    obj_num = random.randint(1, 5)

    for i in range(obj_num):
        cls = np.random.choice(clss)
        obj_idx = random.randint(0, len(objects[cls]) - 1)
        obj = objects[cls][obj_idx]

        jit_factor = random.uniform(*copypaste_scale)
        obj_h = min(obj.shape[0] * jit_factor, img_h)
        obj_w = min(obj.shape[1] * jit_factor, img_w)
        obj = cv2.resize(
            obj,
            (int(obj_w), int(obj_h)),
            interpolation=cv2.INTER_LINEAR,
        )

        blank_x = img_w - obj.shape[1]
        blank_y = img_h - obj.shape[0]
        x1 = random.randint(0, int(blank_x))
        y1 = random.randint(0, int(blank_y))
        x2 = x1 + obj.shape[1]
        y2 = y1 + obj.shape[0]

        new_label = np.array([x1, y1, x2, y2, cls])
        ioa = bbox_ioa(new_label[:4], labels[:, :4])
        if ioa.max() < 0.2:
            img[y1:y2, x1:x2] = 0.0 * img[y1:y2, x1:x2] + 1.0 * obj
            labels = np.vstack((labels, new_label))

    return img.astype(np.uint8), labels
