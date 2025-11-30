import torch
import torch.nn as nn

__all__ = ['AverageMeter', 'evaluate']


class AverageMeter(object):
    r"""Computes and stores the average and current value
       Imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """
    def __init__(self, name):
        self.reset()
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.name = name

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __repr__(self):
        return f"==> For {self.name}: sum={self.sum}; avg={self.avg}"

class CosineSimilarity(nn.Module):
    def forward(self, tensor1, tensor2):
        norm_tensor_1 = tensor1 / tensor1.norm(dim=-1, keep_dim=True)
        norm_tensor_2 = tensor2 / tensor2.norm(dim=-1, keep_dim=True)
        return (norm_tensor_1*norm_tensor_2).sum(dim=-1)

def denormalize(x, min=40, max=170):
    return x * (max-min) + min

def evaluate(preds, gt, norm_BP=False, norm_scope=[40, 170]):
    with torch.no_grad():
        if norm_BP:
            preds = denormalize(preds, norm_scope[0], norm_scope[1])
            gt = denormalize(gt, norm_scope[0], norm_scope[1])
        diff = preds - gt
        mae = diff.abs().mean()
        mean_error = diff.mean()
        var_error = torch.var(diff, unbiased=False)
        rho = torch.cosine_similarity(preds-preds.mean(dim=-1, keepdim=True), gt-gt.mean(dim=-1, keepdim=True), dim=-1).mean()
        return mae, mean_error, var_error, rho

def cal_var(preds_list, gt_list, norm_BP=False, norm_scope=[40, 170]):
    with torch.no_grad():
        if norm_BP:
            preds_list = denormalize(preds_list, norm_scope[0], norm_scope[1])
            gt_list = denormalize(gt_list, norm_scope[0], norm_scope[1])
        diff = preds_list - gt_list
        var = torch.var(diff, unbiased=False)
    return var


def cal_sbp_dbp(preds_list, gt_list, norm_BP=False, norm_scope=[40, 170]):
    with torch.no_grad():
        if norm_BP:
            preds_list = denormalize(preds_list, norm_scope[0], norm_scope[1])
            gt_list = denormalize(gt_list, norm_scope[0], norm_scope[1])
        sbp_p, _ = torch.max(preds_list, dim=-1)
        dbp_p, _ = torch.min(preds_list, dim=-1)
        sbp_g, _ = torch.max(gt_list, dim=-1)
        dbp_g, _ = torch.min(gt_list, dim=-1)
        mae_sbp, mean_error_sbp, var_error_sbp, rho_sbp = evaluate(sbp_p, sbp_g, norm_BP, norm_scope)
        rho_sbp = torch.cosine_similarity(sbp_p.squeeze(), sbp_g.squeeze(), dim=0)
        mae_dbp, mean_error_dbp, var_error_dbp, rho_dbp = evaluate(dbp_p, dbp_g, norm_BP, norm_scope)
        rho_dbp = torch.cosine_similarity(dbp_p.squeeze(), dbp_g.squeeze(), dim=0)
        return [mae_sbp, mean_error_sbp, var_error_sbp, rho_sbp], [mae_dbp, mean_error_dbp, var_error_dbp, rho_dbp]




