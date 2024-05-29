import torch


def transform_sampling(probs, pi, xi):
    cdf = torch.cumsum(torch.gather(probs, 1, pi), 1)
    tokens = torch.gather(pi, 1, torch.searchsorted(cdf, xi))
    return tokens, torch.gather(probs, 1, pi)[tokens]
