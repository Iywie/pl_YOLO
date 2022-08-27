import torch


class YOLOXDecoder:
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

    def __call__(self, inputs):

        with torch.no_grad():
            preds = []
            batch_size = inputs[0].shape[0]
            n_ch = 4 + 1 + self.num_classes  # the channel of one ground truth prediction

            # flatten pyramid predictions
            for i in range(len(inputs)):
                pred = inputs[i]
                h, w = pred.shape[-2:]
                # Three steps to localize predictions: grid, shifts of x and y, grid with stride
                if self.grids[i].shape[2:4] != pred.shape[2:4]:
                    yv, xv = torch.meshgrid([torch.arange(h), torch.arange(w)], indexing='ij')
                    grid = torch.stack((xv, yv), 2).view(1, 1, h, w, 2).type_as(pred)
                    grid = grid.view(1, -1, 2)
                    self.grids[i] = grid
                else:
                    grid = self.grids[i]
                # pred: [batch_size, h * w, n_ch]
                pred = pred.view(batch_size, self.n_anchors, n_ch, h, w)
                pred = pred.permute(0, 1, 3, 4, 2).reshape(batch_size, self.n_anchors * h * w, -1)
                # (x,y + offset) * stride
                pred[..., :2] = (pred[..., :2] + grid) * self.strides[i]
                # The predictions of w and y are logs * stride
                pred[..., 2:4] = torch.exp(pred[..., 2:4]) * self.strides[i]
                preds.append(pred)

            # preds: [batch_size, all predictions, n_ch]
            predictions = torch.cat(preds, 1)
            predictions[..., 4] = predictions[..., 4].sigmoid()
            predictions[..., 5:] = predictions[..., 5:].sigmoid()

            # from (cx,cy,w,h) to (x1,y1,x2,y2)
            box_corner = predictions.new(predictions.shape)
            box_corner = box_corner[:, :, 0:4]
            box_corner[:, :, 0] = predictions[:, :, 0] - predictions[:, :, 2] / 2
            box_corner[:, :, 1] = predictions[:, :, 1] - predictions[:, :, 3] / 2
            box_corner[:, :, 2] = predictions[:, :, 0] + predictions[:, :, 2] / 2
            box_corner[:, :, 3] = predictions[:, :, 1] + predictions[:, :, 3] / 2
            predictions[:, :, :4] = box_corner[:, :, :4]

        return predictions
