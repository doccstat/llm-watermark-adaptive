from numpy import inf, log
from torch.linalg import norm

from numpy import sum as np_sum
from torch import cumsum, from_numpy, mean, pow as torch_pow, searchsorted, tensor, transpose, arange, cat, zeros_like, gather
from scipy.optimize import minimize_scalar


def transform_score(tokens, xi, vocab_size, probs=None, ntps=None):
    metrics = tokens - xi.squeeze()
    return torch_pow(norm(metrics, ord=1), 1)


def transform_adaptive(tokens, xi, vocab_size, probs, ntps=None, shrinkage=1.0):
    # Shrinkage
    probs = shrinkage * probs + (1 - shrinkage) * 0.5

    metrics = (1 / probs - 1) * (tokens - xi.squeeze())
    return torch_pow(norm(metrics, ord=1), 1)


def its_score(tokens, xi, vocab_size, probs=None, ntps=None):
    # tokens = tokens.float() / vocab_size
    metrics = (tokens - 0.5) * (xi.squeeze() - 0.5)
    return -mean(metrics)


def its_weighted_score(tokens, xi, vocab_size, probs, ntps=None, shrinkage=1.0):
    # Shrinkage
    probs = shrinkage * probs + (1 - shrinkage) * 0.5

    metrics = (1 / probs - 1) * (tokens - 0.5) * (xi.squeeze() - 0.5)
    return -mean(metrics)


def its_adaptive(tokens, xi, vocab_size, probs, ntps, shrinkage=1.0):
    """
    Maximizes the sum(log((1 - epsilon) / probs + epsilon)) with respect to epsilon
    using SciPy's optimization functions.

    Parameters:
    - tokens: Tensor (unused)
    - xi: Tensor (unused)
    - probs: Tensor of probabilities
    - ntps: Tensor of NTPs.
    - shrinkage: Float, shrinkage parameter

    Returns:
    - max_obj: The maximized sum(log((1 - epsilon) / probs + epsilon))
    """
    # Apply shrinkage to probs
    probs = shrinkage * probs + (1 - shrinkage) * 0.5

    # Convert probs to NumPy array for SciPy
    probs_np = probs.detach().numpy()

    # `cdf` is of shape (len(tokens), vocab_size)
    # cdf = cumsum(transpose(ntps, 0, 1), 1)

    # Method 1: Extract xi's corresponding tokens and compare.
    # indices = searchsorted(cdf, xi, right=False)
    # indices = indices.float() / ntps.shape[0]
    # indices = indices.squeeze()
    # indicators = (indices == tokens).detach().numpy()

    # Method 2: Extract token's corresponding probabilities and compare with xi.
    # cdf_lower = cat([zeros_like(cdf[:, 0]).unsqueeze(1), cdf[:, :-1]], 1)
    # cdf_upper = cdf

    # The shape of `xi` is (len(tokens)).
    xi = xi.squeeze()
    indicators = (ntps[0] < xi) & (xi <= ntps[1])
    # indicators = gather(indicators, 1, tokens.unsqueeze(1)).squeeze().detach().numpy()
    indicators = indicators.squeeze().detach().numpy()

    # Define the objective function to minimize (negative for maximization)
    def objective(epsilon):
        # Ensure epsilon stays within (0,1) to avoid log issues
        if epsilon <= 0 or epsilon >= 1:
            return inf
        return -np_sum(log(((1 - epsilon) / probs_np) * indicators + epsilon))

    # Use minimize_scalar with bounds and method
    res = minimize_scalar(objective,
                          bounds=(1e-6, 1 - 1e-6),
                          method='bounded',
                          options={
                              'xatol': 1e-8,
                              'maxiter': 100
                          })

    return tensor(res.fun, dtype=probs.dtype, device=probs.device)
