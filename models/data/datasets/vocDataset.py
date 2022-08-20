import os
import cv2
import numpy as np
import xml.etree.ElementTree as ET
from torch.utils.data.dataset import Dataset


class VOCDataset(Dataset):

    """
    VOC Detection Dataset Object
    input is image, target is annotation
    Args:
        data_dir (string): filepath to VOCdevkit folder.
        image_set (string): imageset to use (eg. 'train', 'val', 'test')
        transform (callable, optional): transformation to perform on the
            input image
        target_transform (callable, optional): transformation to perform on the
            target `annotation`
            (eg: take in caption string, return tensor of word indices)
        dataset_name (string, optional): which dataset to load
            (default: 'VOC2007')
    """

    def __init__(
        self,
        data_dir,
        img_subdir,
        ann_subdir,
        image_set,
        img_size,
        classes,
        preprocess=None,
        cache=False,
    ):
        super().__init__()
        self.data_dir = data_dir
        self._imgpath = os.path.join(data_dir, img_subdir)
        self._annopath = os.path.join(data_dir, ann_subdir)
        self.img_size = img_size
        self.preprocess = preprocess
        self.ids = list()
        for line in open(os.path.join(self.data_dir, "ImageSets", image_set + ".txt")):
            self.ids.append(line.strip())
        class_index = [i + 1 for i in range(len(classes))]
        self.class_to_ind = dict(zip(classes, class_index))

        self.keep_difficult = True
        self.annotations = self._load_voc_annotations()
        self.imgs = None
        if cache:
            self._cache_images()

    def __len__(self):
        return len(self.ids)

    def _load_voc_annotations(self):
        return [self.load_anno_from_ids(_ids) for _ids in range(len(self.ids))]

    def load_anno_from_ids(self, index):
        img_id = self.ids[index]
        xml_path = os.path.join(self._annopath, f'{img_id}.xml')
        # xml labels
        target = ET.parse(xml_path).getroot()
        res = np.empty((0, 5))
        for obj in target.iter("object"):
            difficult = obj.find("difficult")
            if difficult is not None:
                difficult = int(difficult.text) == 1
            else:
                difficult = False
            if not self.keep_difficult and difficult:
                continue
            name = obj.find("name").text.strip()
            bbox = obj.find("bndbox")

            pts = ["xmin", "ymin", "xmax", "ymax"]
            bndbox = []
            for i, pt in enumerate(pts):
                cur_pt = int(float(bbox.find(pt).text)) - 1
                # scale height or width
                # cur_pt = cur_pt / width if i % 2 == 0 else cur_pt / height
                bndbox.append(cur_pt)
            label_idx = self.class_to_ind[name]
            bndbox.append(label_idx)
            res = np.vstack((res, bndbox))  # [xmin, ymin, xmax, ymax, label_ind]

        width = int(target.find("size").find("width").text)
        height = int(target.find("size").find("height").text)
        img_hw = (height, width)

        r = min(self.img_size[0] / height, self.img_size[1] / width)
        res[:, :4] *= r

        resized_hw = (int(height * r), int(width * r))

        file_name = target.find('filename').text

        return res, img_hw, resized_hw, file_name

    def load_anno(self, index):
        return self.annotations[index][0]

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
        img_id = self.ids[index]
        img_path = os.path.join(self._imgpath, f'{img_id}.jpg')
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        assert img is not None, f'The problem image is {img_path}'
        return img

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
