import torch


def ems_score(tokens, xi, probs=None):
    xi_samp = torch.gather(xi, -1, tokens.unsqueeze(-1)).squeeze()
    return -torch.mean(torch.log(xi_samp))


def ems_adaptive(
        tokens, xi, probs, percentage=1.0, threshold=0.0, add_constant=0.0
):
    xi_samp = torch.gather(xi, -1, tokens.unsqueeze(-1)).squeeze()
    probs = torch.where(probs <= threshold, threshold, probs)
    probs = torch.where(probs + add_constant >= 1.0, 1.0, probs + add_constant)
    metrics = (1 / probs - 1) * torch.log(xi_samp)
    return -torch.mean(metrics[int(-percentage * len(metrics)):])
