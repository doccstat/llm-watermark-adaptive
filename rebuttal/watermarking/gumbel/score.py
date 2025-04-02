from torch import gather, log, mean, sum as torch_sum, where

from watermarking.gumbel.gumbel_levenshtein import gumbel_levenshtein


def gumbel_score(tokens, xi):
    xi_samp = gather(xi, -1, tokens.unsqueeze(-1)).squeeze()
    return -torch_sum(log(1 / (1 - xi_samp)))


def gumbel_edit_score(tokens, xi, gamma):
    return gumbel_levenshtein(tokens.numpy(), xi.numpy(), gamma)


def ems_score(tokens, xi, probs=None):
    xi_samp = gather(xi, -1, tokens.unsqueeze(-1)).squeeze()
    return -mean(log(xi_samp))


def ems_adaptive(tokens,
                 xi,
                 probs,
                 percentage=1.0,
                 threshold=0.0,
                 add_constant=0.0,
                 shrinkage=1.0):
    """
    Calculate the score for exponential minimum sampling watermarking schemes.

    Args:
        tokens (torch.Tensor): The tokens containing the watermarks.
        xi (torch.Tensor): The watermarking keys.
        probs (torch.Tensor): The probabilities associated with each token.
        percentage (float, optional): The percentage of scores used in
            calculation counting backwards. Defaults to 1.0.
        threshold (float, optional): The threshold value for winsorization.
            Defaults to 0.0.
        add_constant (float, optional): The constant value to add to
            probabilities. Defaults to 0.0.

    Returns:
        torch.Tensor: The negative mean of the calculated metrics.

    """
    xi_samp = gather(xi, -1, tokens.unsqueeze(-1)).squeeze()

    # Winsorization
    probs = where(probs <= threshold, threshold, probs)

    # Shrinkage
    probs = shrinkage * probs + (1 - shrinkage) * 0.5

    # Add constant
    probs = where(probs + add_constant >= 1.0, probs, probs + add_constant)
    metrics = (1 / probs - 1) * log(xi_samp)
    return -mean(metrics[int(-percentage * len(metrics)):])
