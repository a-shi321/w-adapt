import torch
import torch.nn as  nn
import  numpy as np


def reverse_sigmoid(y):
    return torch.log(y / (1.0 - y + 1e-10) + 1e-10)


def get_source_share_weight(domain_out, before_softmax):

    after_softmax=before_softmax

    domain_logit = reverse_sigmoid(domain_out)
    domain_logit = domain_logit / 1.0
    domain_out = nn.Sigmoid()(domain_logit)

    entropy = torch.sum(- after_softmax * torch.log(after_softmax + 1e-10), dim=1, keepdim=True)
    entropy_norm = entropy / np.log(after_softmax.size(1))

    weight = entropy_norm - domain_out
    weight = weight.detach()
    return weight


def get_target_share_weight(domain_out, before_softmax):
    return - get_source_share_weight(domain_out, before_softmax)


def normalize(x):
    min_val = x.min()
    max_val = x.max()
    x = (x - min_val) / (max_val - min_val)
    x = x / torch.mean(x)
    return x.detach()