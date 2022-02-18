import torch
import torch.nn as nn


class YOLOv3Decoder(nn.Module):
    def __init__(
            self,
            num_classes,
            n_anchors=1,
            anchors=None,
            strides=None,
    ):
        super().__init__()
        if anchors is None:
            self.anchors = [(10, 13), (16, 30), (33, 23), (30, 61), (62, 45),
                            (59, 119), (116, 90), (156, 198), (373, 326)]
        self.anchors = torch.Tensor(anchors)
        # self.anchors_mask = anchors_mask
        self.n_anchors = n_anchors
        self.num_classes = num_classes
        self.strides = strides
        self.grids = [torch.zeros(1)] * len(strides)

    def forward(self, inputs):
        preds = []
        shifts_x = []
        shifts_y = []
        expanded_strides = []
        batch_size = inputs[0].shape[0]
        n_ch = 4 + 1 + self.num_classes  # the channel of one ground truth prediction.

        for i in range(len(inputs)):
            # Change all inputs to the same channel.
            pred = inputs[i]
            h, w = pred.shape[-2:]

            # Three steps to localize predictions: grid, shifts of x and y, grid with stride
            if self.grids[i].shape[2:4] != pred.shape[2:4]:
                yv, xv = torch.meshgrid([torch.arange(h), torch.arange(w)])
                grid = torch.stack((xv, yv), 2).view(1, 1, h, w, 2).type_as(pred)
                grid = grid.repeat(1, self.n_anchors, 1, 1, 1)
                # if self.n_anchors > 1:
                #     grid = grid.unsqueeze(4)
                #     grid = grid.repeat(1, 1, 1, 1, self.n_anchors, 1)
                # grid = grid.view(1, -1, 2)
                # # grid: [1, n_anchor * h * w, 2]
                self.grids[i] = grid
            else:
                grid = self.grids[i]
            shifts_x.append(grid[:, :, 0])
            shifts_y.append(grid[:, :, 1])
            expanded_strides.append(
                torch.zeros(1, grid.shape[1]).fill_(self.strides[i]).type_as(pred)
            )
            anchor_w = self.anchors[:, 0].view(1, -1, 1, 1).repeat(1, 1, w, h)
            anchor_h = self.anchors[:, 1].view(1, -1, 1, 1).repeat(1, 1, w, h)

            pred = pred.view(batch_size, self.n_anchors, n_ch, h, w)
            # pred = pred.permute(0, 1, 4, 3, 2).reshape(
            #     batch_size, self.n_anchors * h * w, -1
            # )
            pred = pred.permute(0, 1, 4, 3, 2)
            # pred: [batch_size, n_anchor, w, h, n_ch]
            pred[..., :2] = (pred[..., :2].sigmoid() + grid) * self.strides[i]
            # The predictions of w and y are logs
            pred[..., 2] = torch.exp(pred[..., 2]) * anchor_w
            pred[..., 3] = torch.exp(pred[..., 3]) * anchor_h
            preds.append(pred)

        return preds


    # def forward(self, inputs):
    #     preds = []
    #     x_shifts = []
    #     y_shifts = []
    #     expanded_strides = []
    #     maps_h = []
    #     maps_w = []
    #     batch_size = inputs[0].shape[0]
    #     n_ch = 4 + 1 + self.num_classes  # the channel of one ground truth prediction.
    #
    #     for i in range(len(inputs)):
    #         # Change all inputs to the same channel.
    #         pred = inputs[i]
    #         h, w = pred.shape[-2:]
    #         maps_h.append(h)
    #         maps_w.append(w)
    #         # Three steps to localize predictions: grid, shifts of x and y, grid with stride
    #         if self.grids[i].shape[2:4] != pred.shape[2:4]:
    #             yv, xv = torch.meshgrid([torch.arange(h), torch.arange(w)])
    #             grid = torch.stack((xv, yv), 2).view(1, 1, h, w, 2).type_as(pred)
    #             if self.n_anchors > 1:
    #                 grid = grid.unsqueeze(4)
    #                 grid = grid.repeat(1, 1, 1, 1, self.n_anchors, 1)
    #             grid = grid.view(1, -1, 2)
    #             # grid: [1, n_anchor * h * w, 2]
    #             self.grids[i] = grid
    #         else:
    #             grid = self.grids[i]
    #         x_shifts.append(grid[:, :, 0])
    #         y_shifts.append(grid[:, :, 1])
    #         expanded_strides.append(
    #             torch.zeros(1, grid.shape[1]).fill_(self.strides[i]).type_as(pred)
    #         )
    #         anchor_w = self.anchors[:, 0].repeat(h*w)
    #         anchor_h = self.anchors[:, 1].repeat(h*w)
    #
    #         pred = pred.view(batch_size, self.n_anchors, n_ch, h, w)
    #         pred = pred.permute(0, 1, 3, 4, 2).reshape(
    #             batch_size, self.n_anchors * h * w, -1
    #         )
    #         # pred: [batch_size, n_anchor * h * w, n_ch]
    #         pred[..., :2] = (pred[..., :2].sigmoid() + grid) * self.strides[i]
    #         # The predictions of w and y are logs
    #         pred[..., 2] = torch.exp(pred[..., 2]) * anchor_w
    #         pred[..., 3] = torch.exp(pred[..., 3]) * anchor_h
    #         pred[..., 3] = pred[..., 3].sigmoid()
    #         preds.append(pred)
    #
    #     # preds: [batch_size, all predictions, n_ch]
    #     preds = torch.cat(preds, 1)
    #     return preds, maps_h, maps_w