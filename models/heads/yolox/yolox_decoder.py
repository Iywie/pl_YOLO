import torch
import torch.nn as nn


class YOLOXDecoder(nn.Module):
    def __init__(
            self,
            num_classes,
            strides=None,
    ):
        super().__init__()
        self.n_anchors = 1
        self.num_classes = num_classes
        self.strides = strides
        self.grids = [torch.zeros(1)] * len(strides)

    def forward(self, inputs):
        preds = []
        x_shifts = []
        y_shifts = []
        expanded_strides = []
        batch_size = inputs[0].shape[0]
        n_ch = 4 + 1 + self.n_anchors * self.num_classes  # the channel of one ground truth prediction.

        for i in range(len(inputs)):
            # Change all inputs to the same channel.
            pred = inputs[i]
            h, w = pred.shape[-2:]

            # Three steps to localize predictions: grid, shifts of x and y, grid with stride
            if self.grids[i].shape[2:4] != pred.shape[2:4]:
                yv, xv = torch.meshgrid([torch.arange(h), torch.arange(w)])
                grid = torch.stack((xv, yv), 2).view(1, 1, h, w, 2).type_as(pred)
                grid = grid.view(1, -1, 2)
                # grid: [1, h * w, 2]
                self.grids[i] = grid
            else:
                grid = self.grids[i]
            x_shifts.append(grid[:, :, 0])
            y_shifts.append(grid[:, :, 1])
            expanded_strides.append(
                torch.zeros(1, grid.shape[1]).fill_(self.strides[i]).type_as(pred)
            )

            pred = pred.view(batch_size, self.n_anchors, n_ch, h, w)
            pred = pred.permute(0, 1, 3, 4, 2).reshape(
                batch_size, self.n_anchors * h * w, -1
            )
            # pred: [batch_size, h * w, n_ch]
            pred[..., :2] = (pred[..., :2] + grid) * self.strides[i]
            # The predictions of w and y are logs
            pred[..., 2:4] = torch.exp(pred[..., 2:4]) * self.strides[i]
            preds.append(pred)

        # preds: [batch_size, all predictions, n_ch]
        preds = torch.cat(preds, 1)
        if self.training:
            return preds, x_shifts, y_shifts, expanded_strides
        else:
            return preds

