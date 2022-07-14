


class NewCutOut:
    """CutOut operation.

    Randomly drop some regions of image used in
    `Cutout <https://arxiv.org/abs/1708.04552>`_.

    Args:
        n_holes (int | tuple[int, int]): Number of regions to be dropped.
            If it is given as a list, number of holes will be randomly
            selected from the closed interval [`n_holes[0]`, `n_holes[1]`].
        cutout_shape (tuple[int, int] | list[tuple[int, int]]): The candidate
            shape of dropped regions. It can be `tuple[int, int]` to use a
            fixed cutout shape, or `list[tuple[int, int]]` to randomly choose
            shape from the list.
        cutout_ratio (tuple[float, float] | list[tuple[float, float]]): The
            candidate ratio of dropped regions. It can be `tuple[float, float]`
            to use a fixed ratio or `list[tuple[float, float]]` to randomly
            choose ratio from the list. Please note that `cutout_shape`
            and `cutout_ratio` cannot be both given at the same time.
        fill_in (tuple[float, float, float] | tuple[int, int, int]): The value
            of pixel to fill in the dropped regions. Default: (0, 0, 0).
    """

    def __init__(self,
                 n_holes,
                 cutout_shape=None,
                 cutout_ratio=None,
                 mixup=1.0,
                 prob=0.5):

        assert (cutout_shape is None) ^ (cutout_ratio is None), \
            'Either cutout_shape or cutout_ratio should be specified.'
        assert (isinstance(cutout_shape, (list, tuple))
                or isinstance(cutout_ratio, (list, tuple)))
        if isinstance(n_holes, tuple):
            assert len(n_holes) == 2 and 0 <= n_holes[0] < n_holes[1]
        else:
            n_holes = (n_holes, n_holes)
        self.n_holes = n_holes
        self.with_ratio = cutout_ratio is not None
        self.candidates = cutout_ratio if self.with_ratio else cutout_shape
        if not isinstance(self.candidates, list):
            self.candidates = [self.candidates]
        self.mixup = mixup
        self.prob = prob

    def __call__(self, results):
        """Call function to drop some regions of image."""
        if random.uniform(0, 1) > self.prob:
            return results

        img = results['img']
        h, w, c = img.shape
        gt_bboxes = results['gt_bboxes'].astype(int)
        a = gt_bboxes[:, 0] > 0
        b = gt_bboxes[:, 1] > 0
        c = gt_bboxes[:, 2] < w
        d = gt_bboxes[:, 3] < h
        fills = []
        for i in range(len(gt_bboxes)):
            if a[i] and b[i]:
                left = img[gt_bboxes[i, 1]-1:gt_bboxes[i, 3], gt_bboxes[i, 0]-1:gt_bboxes[i, 0]].mean(0)
                fills.append(left)
            if c[i] and b[i]:
                top = img[gt_bboxes[i, 1]-1:gt_bboxes[i, 1], gt_bboxes[i, 0]:gt_bboxes[i, 2]+1].mean(1)
                fills.append(top)
            if c[i] and d[i]:
                right = img[gt_bboxes[i, 1]:gt_bboxes[i, 3]+1, gt_bboxes[i, 2]:gt_bboxes[i, 2]+1].mean(0)
                fills.append(right)
            if a[i] and d[i]:
                bottom = img[gt_bboxes[i, 3]:gt_bboxes[i, 3]+1, gt_bboxes[i, 0] - 1:gt_bboxes[i, 2]].mean(1)
                fills.append(bottom)
        if len(fills) != 0:
            fill_in = np.array(fills).mean(0).reshape(3)
        else:
            fill_in = np.array([114, 114, 114])
        n_holes = np.random.randint(self.n_holes[0], self.n_holes[1] + 1)
        for _ in range(n_holes):
            x1 = np.random.randint(0, w)
            y1 = np.random.randint(0, h)
            index = np.random.randint(0, len(self.candidates))
            if not self.with_ratio:
                cutout_w, cutout_h = self.candidates[index]
            else:
                cutout_w = int(self.candidates[index][0] * w)
                cutout_h = int(self.candidates[index][1] * h)

            x2 = np.clip(x1 + cutout_w, 0, w)
            y2 = np.clip(y1 + cutout_h, 0, h)
            # ioa = bbox_ioa([x1, y1, x2, y2], gt_bboxes)
            # if ioa.max() < 0.1:
            cut = np.ones(img[y1:y2, x1:x2, :].shape) * fill_in
            img[y1:y2, x1:x2, :] = self.mixup * cut + (1 - self.mixup) * img[y1:y2, x1:x2, :]
            results['img'] = img.astype(np.uint8)

        return results