import torch


def ems_score(tokens, xi):
    xi_samp = torch.gather(xi, -1, tokens.unsqueeze(-1)).squeeze()
    return -torch.mean(torch.log(xi_samp))


def ems_adaptive(tokens, xi, probs, percentage=1.0):
    xi_samp = torch.gather(xi, -1, tokens.unsqueeze(-1)).squeeze()
    metrics = (1 / probs - 1) * torch.log(xi_samp)
    return -torch.mean(metrics[int(-percentage * len(metrics)):])
