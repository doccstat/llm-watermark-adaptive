import sys
import numpy as np

import torch


def ems_score(tokens, xi):
    xi_samp = torch.gather(xi, -1, tokens.unsqueeze(-1)).squeeze()
    return -torch.mean(torch.log(xi_samp))


def ems_adaptive(tokens, xi, probs):
    xi_samp = torch.gather(xi, -1, tokens.unsqueeze(-1)).squeeze()
    return -torch.mean((1 / probs - 1) * torch.log(xi_samp))
