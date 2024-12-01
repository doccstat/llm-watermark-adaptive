from torch import argmax, gather


def gumbel_sampling(probs, pi, xi):
    tokens = argmax(xi ** (1/gather(probs, 1, pi)), axis=1).unsqueeze(-1)
    return tokens, gather(gather(probs, 1, pi), 1, tokens)
