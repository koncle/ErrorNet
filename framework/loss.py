import torch.nn as nn
import torch.functional as F
import torch

class BCELoss2D(nn.Module):
    def __init__(self):
        super(BCELoss2D, self).__init__()
        self._bce_loss = nn.BCEWithLogitsLoss()

    def forward(self, logits, labels):
        logits_flatten = logits.view(-1)
        labels_flatten = labels.view(-1)
        return self._bce_loss(logits_flatten, labels_flatten)

def bce_loss(logits, labels):
    return BCELoss2D()(logits, labels)

def dice_loss(pred, labels, is_average=True):
    num = labels.size(0)
    m1 = pred.view(num, -1)
    m2 = labels.view(num, -1)
    intersection = (m1 * m2)
    score = 2 * (intersection.sum(1) + 1) / (m1.sum(1) + m2.sum(1) + 1)

    if is_average:
        return score.sum() /num
    else:
        return score


class SidedBCELoss(nn.Module):
    def __init__(self, pos_weight, weight):
        super(SidedBCELoss, self).__init__()
        self.weight = weight
        self.pos_weight = pos_weight

    def forward(self, inputs, target):
        inputs = inputs.view(-1)
        target = target.view(-1)
        max_val = (-inputs).clamp(min=0)
        if self.pos_weight is None:
            loss = inputs - inputs * target + max_val + ((-max_val).exp() + (-inputs - max_val).exp()).log()
        else:
            log_weight = 1 + (self.pos_weight - 1) * target
            loss = inputs - inputs * target + log_weight * (max_val + ((-max_val).exp() + (-inputs - max_val).exp()).log())
        if self.weight is None:
            return loss.mean()
        else:
            weight_map = self.get_weight_map(inputs, target)
            return (weight_map * loss).mean()

    def get_weight_map(self, x, y):
        x_label = torch.round(torch.sigmoid(x))
        loss_map = x_label - y
        weight_map = torch.ones_like(loss_map)
        # predict 0 to 1
        weight_map[loss_map == 1] = self.weight[0]
        # predict 1 to 0
        weight_map[loss_map == -1] = self.weight[1]

        weight_map = weight_map.detach()
        return weight_map

