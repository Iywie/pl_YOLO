import os
import cv2
import numpy as np
from models.data.datasets.pycocotools.coco import COCO
from torch.utils.data.dataset import Dataset
from models.data.augmentation.background import getBackground


class COCODataset(Dataset):
    """
        COCO dataset initialization. Annotation data are read into memory by COCO API.
        Args:
            data_dir (str): dataset root directory
            name (str): COCO data name (e.g. 'train2017' or 'val2017')
            img_size (tuple): target image size after pre-processing
            preprocess: data augmentation strategy
            cache(bool):
        """

    def __init__(self,
                 data_dir,
                 name,
                 img_size,
                 preprocess=None,
                 cache=False,
                 ):
        super().__init__()
        self.data_dir = data_dir
        self.name = name
        self.img_size = img_size
        self.preprocess = preprocess
        self.json_file = name + ".json"

        self.coco = COCO(os.path.join(self.data_dir, "annotations", self.json_file))
        self.ids = self.coco.getImgIds()
        self.class_ids = sorted(self.coco.getCatIds())
        self.annotations = self._load_coco_annotations()
        self.imgs = None
        if cache:
            self._cache_images()
        else:
            self.imgs = self._load_imgs()
        # Background imgs and blocks
        self.back_blocks, self.back_cls, self.object_cls = getBackground(
            self.imgs, self.annotations, self.class_ids)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        """
        One image / label pair for the given index is picked up and pre-processed.
        Args:
            index (int): data index
        Returns:
            img (numpy.ndarray): pre-processed image
            padded_labels (torch.Tensor): pre-processed label data.
                The shape is: [max_labels, 5]
                each label consists of [class, cx, cy, w, h]
            info_img : tuple of origin h, w
            img_id (int): same as the input index. Used for evaluation.
        """

        # Read annotation from self
        id_ = self.ids[index]
        res, img_hw, resized_info, img_name = self.annotations[index]
        # load image from file
        if self.imgs is not None:
            img = self.imgs[index]
            # img = pad_img[: resized_info[0], : resized_info[1], :].copy()
        else:
            img = self.load_resized_img(index)

        if self.preprocess is not None:
            img, target = self.preprocess(img, res, self.img_size)
        else:
            target = res
        return img, target, img_hw, np.array([id_]), img_name

    def _load_coco_annotations(self):
        return [self.load_anno_from_id(_id) for _id in self.ids]

    def load_anno_from_id(self, id_):
        im_ann = self.coco.loadImgs([id_])[0]  # im_ann: [file_name, height, width, id]
        width = im_ann["width"]
        height = im_ann["height"]
        anno_ids = self.coco.getAnnIds(imgIds=[id_], iscrowd=False)
        annotations = self.coco.loadAnns(anno_ids)
        objs = []
        for obj in annotations:
            x1 = np.max((0, obj["bbox"][0]))
            y1 = np.max((0, obj["bbox"][1]))
            x2 = np.min((width, x1 + np.max((0, obj["bbox"][2]))))
            y2 = np.min((height, y1 + np.max((0, obj["bbox"][3]))))
            if obj["area"] > 0 and x2 >= x1 and y2 >= y1:
                obj["clean_bbox"] = [x1, y1, x2, y2]
                objs.append(obj)

        num_objs = len(objs)

        res = np.zeros((num_objs, 5))
        for ix, obj in enumerate(objs):
            cls = self.class_ids.index(obj["category_id"])
            res[ix, 0:4] = obj["clean_bbox"]
            res[ix, 4] = cls

        r = min(self.img_size[0] / height, self.img_size[1] / width)
        res[:, :4] *= r

        img_hw = (height, width)
        resized_hw = (int(height * r), int(width * r))

        file_name = (
            im_ann["file_name"]
            if "file_name" in im_ann
            else "{:012}".format(id) + ".jpg"
        )

        return res, img_hw, resized_hw, file_name

    def _load_imgs(self):
        return [self.load_resized_img(_id) for _id in self.ids]

    def load_resized_img(self, index):
        img = self.load_image(index)
        r = min(self.img_size[0] / img.shape[0], self.img_size[1] / img.shape[1])
        resized_img = cv2.resize(
            img,
            (int(img.shape[1] * r), int(img.shape[0] * r)),
            interpolation=cv2.INTER_LINEAR,
        ).astype(np.uint8)
        return resized_img

    def load_image(self, index):
        file_name = self.annotations[index][3]
        img_file = os.path.join(self.data_dir, self.name, file_name)
        img = cv2.imread(img_file)
        assert img is not None, f'The problem image is {file_name}'
        return img

    def _cache_images(self):
        print(
            "\n********************************************************************************\n"
            "You are using cached images in RAM to accelerate training.\n"
            "This requires large system RAM. For COCO need 200G+ RAM space.\n"
            "********************************************************************************\n"
        )
        self.imgs = [None] * len(self.annotations)
        from tqdm import tqdm
        from multiprocessing.pool import ThreadPool
        gb = 0
        NUM_THREADs = min(8, os.cpu_count())
        loaded_images = ThreadPool(NUM_THREADs).imap(
            lambda x: self.load_resized_img(x),
            range(len(self.annotations)),
        )
        pbar = tqdm(enumerate(loaded_images), total=len(self.annotations), mininterval=100)
        for k, out in pbar:
            self.imgs[k] = out.copy()
            gb += self.imgs[k].nbytes
            pbar.desc = f'Caching images ({gb / 1E9:.1f}GB)'
        pbar.close()
