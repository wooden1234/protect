import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceBCELoss(nn.Module):
    def __init__(self, smooth=1.):
        super(DiceBCELoss, self).__init__()
        self.smooth = smooth
        self.bce = nn.BCEWithLogitsLoss()  # 注意：适用于模型输出未经 sigmoid 的情况

    def forward(self, pred, target):
        # BCE Loss
        bce_loss = self.bce(pred, target)

        # Dice Loss
        pred = torch.sigmoid(pred)  # 因为 BCEWithLogitsLoss 用的是 logit，这里要 sigmoid
        pred = pred.contiguous()
        target = target.contiguous()

        intersection = (pred * target).sum(dim=2).sum(dim=2)

        dice_loss = 1 - ((2. * intersection + self.smooth) /
                         (pred.sum(dim=2).sum(dim=2) + target.sum(dim=2).sum(dim=2) + self.smooth))

        dice_loss = dice_loss.mean()

        # 混合损失：你也可以改权重，比如 0.7 * bce + 0.3 * dice
        return bce_loss + dice_loss
