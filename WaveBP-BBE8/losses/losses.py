import torch
from torch import nn as nn
from torch.nn import functional as F
IS_HIGH_VERSION = tuple(map(int, torch.__version__.split('+')[0].split('.'))) > (1, 7, 1)
import torch.fft

class Weighted_L2_Corr_Loss(nn.Module):
    def __init__(self, alpha=0.5):
        super(Weighted_L2_Corr_Loss, self).__init__()
        self.alpha = alpha
        print(f"Use Weighted_L2_Corr_Loss, [alpha]={alpha}")
    def forward(self, X, Y, weights=None):
        rho = torch.cosine_similarity(X-X.mean(), Y-Y.mean(), dim=-1).mean()
        cos_loss = 1 - rho
        mse_loss = F.mse_loss(X, Y, reduction='none')
        if weights is not None:
            weights = weights.to(X.device)
            mse_loss = torch.mul(weights, mse_loss)
        mse_loss = mse_loss.mean()
        loss = 1 * mse_loss + self.alpha * cos_loss
        return loss, mse_loss, self.alpha * cos_loss

class HintLoss(nn.Module):
    """Fitnets: hints for thin deep nets, ICLR 2015"""
    def __init__(self):
        super(HintLoss, self).__init__()
        self.crit = nn.MSELoss()

    def forward(self, f_s, f_t):
        loss = self.crit(f_s, f_t)
        return loss