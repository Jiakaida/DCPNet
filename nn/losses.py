
import torch
import torch.nn.functional as F


def structure_loss(pred, mask):
    weight = 1 + 5 * torch.abs(F.avg_pool2d(mask, 31, stride=1, padding=15) - mask)
    bce = F.binary_cross_entropy_with_logits(pred, mask, reduction='none')
    bce = (weight * bce).sum(dim=(2, 3)) / weight.sum(dim=(2, 3))
    probability = pred.sigmoid()
    intersection = (probability * mask * weight).sum(dim=(2, 3))
    union = ((probability + mask) * weight).sum(dim=(2, 3))
    iou = 1 - (intersection + 1) / (union - intersection + 1)
    return (bce + iou).mean()


def detection_loss(result, target, gamma=0.2):
    
    stages = result['side']
    if len(stages) != 4:
        raise ValueError('Eq. (14) requires exactly P1, P2, P3 and P4')
    return structure_loss(result['coarse'], target) + sum(
        gamma * i * structure_loss(prediction, target)
        for i, prediction in enumerate(stages, start=1)
    )
