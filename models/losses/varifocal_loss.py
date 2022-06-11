import torch.nn.functional as F


def varifocal_loss(pred_score, gt_score, label, alpha=0.75, gamma=2.0):
    weight = alpha * pred_score.pow(gamma) * (1 - label) + gt_score * label
    loss = F.binary_cross_entropy(
        pred_score, gt_score, weight=weight, reduction='sum')
    return loss
