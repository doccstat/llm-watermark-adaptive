from torch import cumsum, gather, searchsorted


def transform_sampling(probs, pi, xi):
    cdf = cumsum(gather(probs, 1, pi), 1)
    tokens = gather(pi, 1, searchsorted(cdf, xi))
    return tokens, gather(gather(probs, 1, pi), 1, tokens)
