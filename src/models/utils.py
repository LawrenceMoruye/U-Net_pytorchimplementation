import torch
from torch import nn,optim
import torch.nn.functional as F

def dice_loss(input, target):
    input = torch.sigmoid(input)
    smooth = 1.0
    iflat = input.view(-1)
    tflat = target.view(-1)
    intersection = (iflat * tflat).sum()
    return ((2.0 * intersection + smooth) / (iflat.sum() + tflat.sum() + smooth))


class MixedLoss(nn.Module):
    def __init__(self, alpha=None, gamma=None):
        super().__init__()

    def forward(self, input, target):
        loss = -torch.log(dice_loss(input, target))

        return loss.mean()