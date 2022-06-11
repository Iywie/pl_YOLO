import random
import numpy as np
from models.utils.bbox import bbox_ioa


def cutouta(img, labels, n_hole, cutout_ratio, mixup, ioa_thre):
    # Applies img cutout augmentation https://arxiv.org/abs/1708.04552
    h, w = img.shape[:2]
    n_label = len(labels)

    if n_label == 0:
        return img.astype(np.uint8)

    a = labels[:, 0] > 0
    b = labels[:, 1] > 0
    c = labels[:, 2] < w
    d = labels[:, 3] < h
    fills = []
    for i in range(n_label):
        if a[i] and b[i]:
            left = img[labels[i, 1] - 1:labels[i, 3], labels[i, 0] - 1:labels[i, 0]].mean(0)
            fills.append(left)
        if c[i] and b[i]:
            top = img[labels[i, 1] - 1:labels[i, 1], labels[i, 0]:labels[i, 2] + 1].mean(1)
            fills.append(top)
        if c[i] and d[i]:
            right = img[labels[i, 1]:labels[i, 3] + 1, labels[i, 2]:labels[i, 2] + 1].mean(0)
            fills.append(right)
        if a[i] and d[i]:
            bottom = img[labels[i, 3]:labels[i, 3] + 1, labels[i, 0] - 1:labels[i, 2]].mean(1)
            fills.append(bottom)

    if len(fills) != 0:
        fill_in = np.array(fills).mean(0).reshape(3)
    else:
        fill_in = np.array([114, 114, 114])

    n_hole = np.random.randint(n_hole[0], n_hole[1] + 1)

    for _ in range(n_hole):
        x1 = np.random.randint(0, w)
        y1 = np.random.randint(0, h)
        index = np.random.randint(0, len(cutout_ratio))

        cutout_w, cutout_h = cutout_ratio[index]

        x2 = np.clip(x1 + cutout_w, 0, w)
        y2 = np.clip(y1 + cutout_h, 0, h)
        ioa = bbox_ioa([x1, y1, x2, y2], labels[:, :4])
        if ioa.max() < ioa_thre:
            cut = np.ones(img[y1:y2, x1:x2, :].shape) * fill_in
            img[y1:y2, x1:x2, :] = mixup * cut + (1 - mixup) * img[y1:y2, x1:x2, :]

    return img.astype(np.uint8)
