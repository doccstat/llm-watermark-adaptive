import torch


def ems_score(tokens, xi, probs=None):
    xi_samp = torch.gather(xi, -1, tokens.unsqueeze(-1)).squeeze()
    return -torch.mean(torch.log(xi_samp))


def ems_adaptive(
        tokens, xi, probs, percentage=1.0, threshold=0.0, add_constant=0.0,
        shrinkage=1.0
):
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
    xi_samp = torch.gather(xi, -1, tokens.unsqueeze(-1)).squeeze()

    # Winsorization
    probs = torch.where(probs <= threshold, threshold, probs)

    # Shrinkage
    probs = shrinkage * probs + (1 - shrinkage) * 0.5

    # Add constant
    probs = torch.where(probs + add_constant >= 1.0,
                        probs, probs + add_constant)
    metrics = (1 / probs - 1) * torch.log(xi_samp)
    return -torch.mean(metrics[int(-percentage * len(metrics)):])
