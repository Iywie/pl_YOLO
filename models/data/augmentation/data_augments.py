import cv2
import random
import numpy as np


class TrainTransform:
    def __init__(self, max_labels=50, flip_prob=0.5, hsv_prob=1.0):
        self.max_labels = max_labels
        self.flip_prob = flip_prob
        self.hsv_prob = hsv_prob

    def __call__(self, image, targets, input_dim):

        if len(targets) == 0:
            targets = np.zeros((self.max_labels, 5), dtype=np.float32)
            image, r_o = preproc(image, input_dim)
            return image, targets

        image_process = image.copy()
        targets_process = targets.copy()

        if random.random() < self.hsv_prob:
            augment_hsv(image_process)
        if random.random() < self.flip_prob:
            image_process, targets_process[:, :4] = _mirror(image_process, targets_process[:, :4])
        image_process, r = preproc(image_process, input_dim)

        # boxes [xyxy] 2 [cx,cy,w,h]
        targets_process[:, :4] = xyxy2cxcywh(targets_process[:, :4])
        targets_process[:, :4] *= r

        mask_b = np.minimum(targets_process[:, 2], targets_process[:, 3]) > 1
        targets_process = targets_process[mask_b]

        if len(targets_process) == 0:
            image_process, r_o = preproc(image, input_dim)
            targets_process = targets
            targets_process[:, :4] = r_o * targets_process[:, :4]
            targets_process[:, :4] = xyxy2cxcywh(targets_process[:, :4])

        label_process = np.expand_dims(targets_process[:, 4], 1)

        targets = np.hstack((label_process, targets_process[:, :4]))
        padded_labels = np.zeros((self.max_labels, 5))
        padded_labels[range(len(targets))[: self.max_labels]] = targets[: self.max_labels]
        padded_labels = np.ascontiguousarray(padded_labels, dtype=np.float32)
        return image_process, padded_labels


class ValTransform:
    """
    Defines the transformations that should be applied to test PIL image
    for input into the network

    dimension -> tensorize -> color adj

    Arguments:
        swap ((int,int,int)): final order of channels

    Returns:
        transform (transform) : callable transform to be applied to test/val
        data
    """

    def __init__(self, swap=(2, 0, 1), legacy=False, max_labels=50):
        self.swap = swap
        self.legacy = legacy
        self.max_labels = max_labels

    # assume input is cv2 img for now
    def __call__(self, img, targets, input_size):
        img, _ = preproc(img, input_size, self.swap)
        if self.legacy:
            img = img[::-1, :, :].copy()
            img /= 255.0
            img -= np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
            img /= np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)

        boxes = targets[:, :4].copy()
        labels = targets[:, 4].copy()

        boxes = xyxy2cxcywh(boxes)
        labels = np.expand_dims(labels, 1)
        targets_t = np.hstack((labels, boxes))
        padded_labels = np.zeros((self.max_labels, 5))
        padded_labels[range(len(targets_t))[:self.max_labels]] = targets_t[:self.max_labels]
        return img, padded_labels


def preproc(img, input_size, swap=(2, 0, 1)):
    if len(img.shape) == 3:
        padded_img = np.ones((input_size[0], input_size[1], 3), dtype=np.uint8) * 114
    else:
        padded_img = np.ones(input_size, dtype=np.uint8) * 114

    r = min(input_size[0] / img.shape[0], input_size[1] / img.shape[1])
    resized_img = cv2.resize(
        img,
        (int(img.shape[1] * r), int(img.shape[0] * r)),
        interpolation=cv2.INTER_LINEAR,
    ).astype(np.uint8)
    padded_img[: int(img.shape[0] * r), : int(img.shape[1] * r)] = resized_img

    padded_img = padded_img.transpose(swap)
    padded_img = np.ascontiguousarray(padded_img, dtype=np.float32)
    return padded_img, r


def augment_hsv(img, hgain=0.015, sgain=0.7, vgain=0.4):
    r = np.random.uniform(-1, 1, 3) * [hgain, sgain, vgain] + 1  # random gains
    hue, sat, val = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
    dtype = img.dtype  # uint8

    x = np.arange(0, 256, dtype=np.int16)
    lut_hue = ((x * r[0]) % 180).astype(dtype)
    lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
    lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

    img_hsv = cv2.merge(
        (cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val))
    ).astype(dtype)
    cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR, dst=img)  # no return needed


def _mirror(image, boxes):
    _, width, _ = image.shape
    image = image[:, ::-1]
    boxes[:, 0::2] = width - boxes[:, 2::-2]
    return image, boxes


def xyxy2cxcywh(bboxes):
    bboxes[:, 2] = bboxes[:, 2] - bboxes[:, 0]
    bboxes[:, 3] = bboxes[:, 3] - bboxes[:, 1]
    bboxes[:, 0] = bboxes[:, 0] + bboxes[:, 2] * 0.5
    bboxes[:, 1] = bboxes[:, 1] + bboxes[:, 3] * 0.5
    return bboxes

