import torch


class YOLOv5Decoder:
    def __init__(self, num_classes, anchors, strides):
        super(YOLOv5Decoder, self).__init__()
        self.num_classes = num_classes
        self.anchors = torch.tensor(anchors)
        self.strides = strides
        self.grid = [torch.zeros(1)] * len(anchors)  # init grid
        self.anchor_grid = self.anchors.clone().view(len(anchors), 1, -1, 1, 1, 2)

    def __call__(self, inputs):
        """
        :param inputs: a list of feature maps
        :return:
        """
        nl = len(inputs)  # number of layers
        na = len(self.anchors[0])  # number of anchors, every layer have the same number of anchors
        bs = inputs[0].shape[0]
        z = []  # inference output
        for i in range(nl):
            ny = inputs[i].size(2)
            nx = inputs[i].size(3)
            prediction = inputs[i].view(bs, na, 5+self.num_classes, ny, nx) \
                .permute(0, 1, 3, 4, 2).contiguous()
            inputs[i] = prediction
            if self.grid[i].shape[2:4] != prediction.shape[2:4]:
                self.grid[i] = self._make_grid(nx, ny).type_as(prediction)
                self.anchor_grid = self.anchor_grid.type_as(prediction)
            y = prediction.sigmoid()
            y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + self.grid[i]) * self.strides[i]  # xy
            y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
            z.append(y.view(bs, -1, 5+self.num_classes))

        return torch.cat(z, 1)

    @staticmethod
    def _make_grid(nx=20, ny=20):
        yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)])
        return torch.stack((xv, yv), 2).view((1, 1, ny, nx, 2)).float()
