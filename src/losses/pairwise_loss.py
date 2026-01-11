import torch
import torch.nn.functional as F

def pairwise_logistic_loss(s_pos, s_neg):
    """
    s_pos, s_neg: (B,)
    """
    return -F.logsigmoid(s_pos - s_neg).mean()