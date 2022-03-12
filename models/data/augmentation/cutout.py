import random
import numpy as np
from models.utils.bbox import bbox_ioa


def cutout(image, labels, background=None):
    # Applies image cutout augmentation https://arxiv.org/abs/1708.04552
    h, w = image.shape[:2]
    # create random masks
    scales = [0.5] * 1 + [0.25] * 2 + [0.125] * 4 + [0.0625] * 8 + [0.03125] * 16  # image size fraction
    for s in scales:
        mask_h = random.randint(1, int(h * s))
        mask_w = random.randint(1, int(w * s))

        # box
        xmin = max(0, random.randint(0, w) - mask_w // 2)
        ymin = max(0, random.randint(0, h) - mask_h // 2)
        xmax = min(w, xmin + mask_w)
        ymax = min(h, ymin + mask_h)

        if background is not None:
            block_i = random.randint(0, len(background) - 1)
            block_h, block_w, _ = background[block_i].shape
            block_h = min(block_h, ymax - ymin)
            block_w = min(block_w, xmax - xmin)

            if len(labels) and s > 0.03:
                box = np.array([xmin, ymin, xmin + block_w, ymin + block_h], dtype=np.float32)
                ioa = bbox_ioa(box, labels[:, :4])  # intersection over area
                if ioa.max() > 0.3:
                    continue
            image[ymin:ymin + block_h, xmin:xmin + block_w] = \
                background[block_i][:block_h, :block_w] * 0.8 + image[ymin:ymin + block_h, xmin:xmin + block_w] * 0.2

        else:
            image[ymin:ymax, xmin:xmax] = [random.randint(64, 191) for _ in range(3)]
            # return unobscured labels
            if len(labels) and s > 0.03:
                box = np.array([xmin, ymin, xmax, ymax], dtype=np.float32)
                ioa = bbox_ioa(box, labels[:, :4])  # intersection over area
                labels = labels[ioa < 0.6]  # remove >60% obscured labels

    return image, labels
