from torch.linalg import norm

from torch import pow as torch_pow

from watermarking.transform.transform_levenshtein import transform_levenshtein


def transform_score(tokens, xi, probs=None):
    metrics = tokens - xi.squeeze()
    return torch_pow(norm(metrics, ord=1), 1)


def transform_adaptive(tokens, xi, probs, shrinkage=1.0):
    # Shrinkage
    probs = shrinkage * probs + (1 - shrinkage) * 0.5

    metrics = (1 / probs - 1) * (tokens - xi.squeeze())
    return torch_pow(norm(metrics, ord=1), 1)


def transform_edit_score(tokens, xi, gamma=1):
    return transform_levenshtein(tokens.numpy(), xi.squeeze().numpy(), gamma)
