import torch


def gumbel_sampling(probs, pi, xi):
    tokens = torch.argmax(xi ** (1/torch.gather(probs, 1, pi)), axis=1).unsqueeze(-1)
    return tokens, torch.gather(torch.gather(probs, 1, pi), 1, tokens)
