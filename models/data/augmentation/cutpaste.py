import random
import numpy as np
from models.utils.bbox import bbox_ioa


def cutpaste(img, labels, background=None):
    # Applies img cutout augmentation https://arxiv.org/abs/1708.04552
    h, w = img.shape[:2]

    if len(labels):
        clss = labels[:, 4]
        clss = np.unique(clss)
        clss = clss.astype(int)
    else:
        return img.astype(np.uint8)

    num_patches = random.randint(1, 3)
    for i in range(num_patches):
        if len(clss) > 0:
            cls = np.random.choice(clss)
            j = random.randint(0, len(background[cls])-1)
            bg = background[cls][j]
            h_bg, w_bg = bg.shape[:2]

            offset_x = random.randint(0, w - w_bg)
            offset_y = random.randint(0, h - h_bg)
            xmin = offset_x
            ymin = offset_y
            xmax = offset_x + w_bg
            ymax = offset_y + h_bg

            box = np.array([xmin, ymin, xmax, ymax], dtype=np.float32)
            ioa = bbox_ioa(box, labels[:, :4])  # intersection over area
            if ioa.max() < 0.2:
                img[ymin:ymax, xmin:xmax] = 0.5 * img[ymin:ymax, xmin:xmax] + 0.5 * bg
        else:
            return img.astype(np.uint8)
    return img.astype(np.uint8)
