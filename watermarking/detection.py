import torch
import numpy as np
import time
from scipy.stats import cauchy
from scipy.linalg import expm


def quantile_test(
    tokens, vocab_size, n, k, seed, test_stats, ntps
):
    generator = torch.Generator()

    test_results = []

    for test_stat in test_stats:
        generator.manual_seed(int(seed))
        test_result = test_stat(tokens=tokens,
                                n=n,
                                k=k,
                                generator=generator,
                                vocab_size=vocab_size)
        test_results.append(test_result)

    p_vals = np.empty((3, len(test_stats)))
    test_results = np.array(test_results)
    for test_stat_idx, test_stat in enumerate(test_stats):
        # This is a nasty hack to deal with the weights. This has to be corresponding to the
        # regularizations and experiments in the `./4-detect.py` file.
        ntp = None
        if test_stat_idx in [0]:
            ntp = ntps[0]
        elif test_stat_idx in [1]:
            ntp = ntps[1]
        elif test_stat_idx in [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]:
            ntp = ntps[2]
        elif test_stat_idx in [15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27]:
            ntp = ntps[3]
        elif test_stat_idx in [28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40]:
            ntp = ntps[4]
        elif test_stat_idx in [41, 48]:
            ntp = ntps[5]
        elif test_stat_idx in [42, 49]:
            ntp = ntps[6]
        elif test_stat_idx in [43, 50]:
            ntp = ntps[7]
        elif test_stat_idx in [44, 51]:
            ntp = ntps[8]
        elif test_stat_idx in [45, 52]:
            ntp = ntps[9]
        elif test_stat_idx in [46, 53]:
            ntp = ntps[10]
        elif test_stat_idx in [47, 54]:
            ntp = ntps[11]

        if test_stat_idx in [0, 1, 2, 15, 28]:
            ntp = ntp
        elif test_stat_idx in [3, 16, 29]:
            ntp = torch.where(ntp + 0.001 >= 1.0, ntp, ntp + 0.001)
        elif test_stat_idx in [4, 17, 30]:
            ntp = torch.where(ntp + 0.01 >= 1.0, ntp, ntp + 0.01)
        elif test_stat_idx in [5, 18, 31]:
            ntp = torch.where(ntp + 0.1 >= 1.0, ntp, ntp + 0.1)
        elif test_stat_idx in [6, 19, 32]:
            ntp = 0.9 * ntp + (1 - 0.9) * 0.5
        elif test_stat_idx in [7, 20, 33]:
            ntp = 0.8 * ntp + (1 - 0.8) * 0.5
        elif test_stat_idx in [8, 21, 34]:
            ntp = 0.7 * ntp + (1 - 0.7) * 0.5
        elif test_stat_idx in [9, 22, 35]:
            ntp = 0.6 * ntp + (1 - 0.6) * 0.5
        elif test_stat_idx in [10, 23, 36, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54]:
            ntp = 0.5 * ntp + (1 - 0.5) * 0.5
        elif test_stat_idx in [11, 24, 37]:
            ntp = 0.4 * ntp + (1 - 0.4) * 0.5
        elif test_stat_idx in [12, 25, 38]:
            ntp = 0.3 * ntp + (1 - 0.3) * 0.5
        elif test_stat_idx in [13, 26, 39]:
            ntp = 0.2 * ntp + (1 - 0.2) * 0.5
        elif test_stat_idx in [14, 27, 40]:
            ntp = 0.1 * ntp + (1 - 0.1) * 0.5

        ntp = ntp / (1 - ntp) * len(ntp)

        test_statistics_before_combination = test_results[test_stat_idx]
        pvalues_before_combination = np.empty_like(test_statistics_before_combination)
        # for each row in pvalues_before_combination, the ntp passed to phypoexp is the same
        # for each column, the ntp is different and can be obtained from `ntp` array
        k = len(ntp) + 1 - len(test_statistics_before_combination)
        for i in range(len(test_statistics_before_combination)):
            pvalues_before_combination[i] = phypoexp(
                test_statistics_before_combination[i],
                ntp[np.isfinite(ntp)][i:i+k]
            )
        p_vals[0, test_stat_idx] = cauchy_combine(np.reshape(pvalues_before_combination, (1, -1)))
        p_values_column = np.empty((1, test_statistics_before_combination.shape[1]))
        for i in range(test_statistics_before_combination.shape[1]):
            p_values_column[0, i] = cauchy_combine(pvalues_before_combination[:, i])
        p_vals[1, test_stat_idx] = cauchy_combine(p_values_column)
        p_values_row = np.empty((1, test_statistics_before_combination.shape[0]))
        for i in range(test_statistics_before_combination.shape[0]):
            p_values_row[0, i] = cauchy_combine(pvalues_before_combination[i])
        p_vals[2, test_stat_idx] = cauchy_combine(p_values_row)
    return p_vals


def permutation_test(
    tokens, vocab_size, n, k, seed, test_stats, log_file, n_runs=100, max_seed=100000
):
    generator = torch.Generator()

    test_results = []

    for test_stat in test_stats:
        generator.manual_seed(int(seed))
        test_result = test_stat(tokens=tokens,
                                n=n,
                                k=k,
                                generator=generator,
                                vocab_size=vocab_size)
        test_results.append(test_result)

    test_results = np.array(test_results)
    null_results = []
    t0 = time.time()
    log_file.write(f'Begin {n_runs} permutation tests\n')
    log_file.flush()
    for run in range(n_runs):
        if run % 100 == 0:
            log_file.write(f'Run {run} (t = {time.time()-t0} seconds)\n')
            log_file.flush()
        null_results.append([])

        seed = torch.randint(high=max_seed, size=(1,)).item()
        for test_stat in test_stats:
            generator.manual_seed(int(seed))
            null_result = test_stat(tokens=tokens,
                                    n=n,
                                    k=k,
                                    generator=generator,
                                    vocab_size=vocab_size,
                                    null=True)
            null_results[-1].append(null_result)
    null_results = np.array(null_results)

    return (np.sum(null_results <= test_results, axis=0) + 1.0) / (n_runs + 1.0)


def phi(
        tokens, n, k, generator, key_func, vocab_size, dist, empty_probs,
        null=False, normalize=False, asis=True
):
    if null:
        tokens = torch.unique(torch.asarray(
            tokens), return_inverse=True, sorted=False)[1]
        eff_vocab_size = torch.max(tokens) + 1
    else:
        eff_vocab_size = vocab_size

    xi, pi = key_func(generator, n, vocab_size, eff_vocab_size)
    tokens = torch.argsort(pi)[tokens]
    if normalize:
        tokens = tokens.float() / vocab_size

    A = adjacency(tokens, xi, dist, k, empty_probs)
    if asis:
        return A
    closest = torch.min(A, axis=1)[0]

    return torch.min(closest)


def adjacency(tokens, xi, dist, k, empty_probs):
    m = len(tokens)
    n = len(xi)

    A = torch.empty(size=(m-(k-1), n))
    for i in range(m-(k-1)):
        for j in range(n):
            A[i][j] = dist(tokens[i:i+k], xi[(j+torch.arange(k)) % n], empty_probs[i:i+k])

    return A


def phypoexp(x, rate, lower_tail=True, log_p=False, tailarea=False):
    """
    Compute the CDF or survival function for the hypo-exponential distribution
    using the matrix exponential approach. This method correctly handles
    non-unique (repeated) rate parameters.

    Parameters
    ----------
    x : float or array_like
        Points at which to evaluate the CDF or survival function.
    rate : array_like
        Rates of the exponential phases.
    lower_tail : bool, optional
        If True, compute P(X <= x). If False, compute P(X > x).
    log_p : bool, optional
        If True, return the logarithm of the probability.
    tailarea : bool, optional
        If True, compute the survival function P(X > x) regardless of 'lower_tail'.

    Returns
    -------
    probabilities : float or ndarray
        The CDF or survival function evaluated at x.
    """
    x_original = x  # Preserve original input for shape
    x = np.atleast_1d(x).astype(np.float64)
    rate = np.asarray(rate, dtype=np.float64)
    n = len(rate)

    # Construct the generator matrix Q with an absorbing state
    Q = np.zeros((n + 1, n + 1), dtype=np.float64)

    for i in range(n):
        Q[i, i] = -rate[i]
        Q[i, i + 1] = rate[i]
    Q[-1, -1] = 0.0  # Absorbing state

    # Initial state vector: all probability in the first phase
    v = np.zeros(n + 1)
    v[0] = 1.0

    # Compute e^{Qx} for each x
    probabilities = []
    for xi in x:
        eQt = expm(Q * xi)
        p_absorbed = eQt[0, -1]  # Probability of being absorbed by time xi
        if tailarea:
            P = p_absorbed  # Survival function
        else:
            P = p_absorbed  # CDF
            if not lower_tail:
                P = 1.0 - P  # Invert the tail
        if log_p:
            P = np.log(P) if P > 0 else -np.inf
        probabilities.append(P)

    probabilities = np.array(probabilities)

    # Return scalar if input was scalar
    if np.isscalar(x_original):
        return probabilities[0]
    else:
        return probabilities


def cauchy_combine(pvalues, weights=None):
    """
    Combine p-values using the Cauchy combination method.

    Parameters:
    ----------
    pvalues : array-like
        A 2D array (matrix) of p-values. Each row corresponds to a gene, and each column corresponds to a p-value from different tests.
    weights : array-like, optional
        A 2D array of weights with the same shape as `pvalues`. The weights for each row should sum to 1.
        If not provided, equal weights are assumed.

    Returns:
    -------
    numpy.ndarray
        A 1D array of combined p-values for each gene.

    Raises:
    ------
    ValueError
        If the dimensions of `weights` do not match those of `pvalues`.
    """
    # Convert pvalues to a NumPy array and ensure it's 2D
    pvalues = np.asarray(pvalues)
    if pvalues.ndim != 2:
        pvalues = np.atleast_2d(pvalues)

    # Replace p-values of exactly 0 with a small value to avoid infinity in tan
    pvalues[pvalues == 0] = 5.55e-17

    # Replace p-values where (1 - pvalue) < 1e-3 with 0.99 to avoid extreme values
    pvalues[(1 - pvalues) < 1e-3] = 0.99

    num_genes, num_pvals = pvalues.shape

    # If weights are not provided, use equal weights
    if weights is None:
        weights = np.full((num_genes, num_pvals), 1.0 / num_pvals)
    else:
        weights = np.asarray(weights)
        if weights.shape != pvalues.shape:
            raise ValueError("The dimensions of weights do not match those of pvalues.")

    # Compute Cauchy statistics
    Cstat = np.tan((0.5 - pvalues) * np.pi)

    # Apply weights to the Cauchy statistics
    wCstat = weights * Cstat

    # Sum the weighted Cauchy statistics for each gene
    Cbar = np.sum(wCstat, axis=1)

    # Compute combined p-values using the Cauchy CDF
    combined_pval = 1.0 - cauchy.cdf(Cbar)

    # Replace combined p-values that are <= 0 with a small value
    combined_pval[combined_pval <= 0] = 5.55e-17

    return combined_pval


if __name__ == "__main__":
    # Example rate parameters with non-unique rates
    rates = [1.0, 2.0, 2.0, 3.0]  # lambda=2.0 has multiplicity 2

    # Quantiles at which to evaluate the CDF
    quantiles = np.array([0.5, 1.0, 1.5, 2.0])

    # Compute the CDF
    print("CDF Values:", phypoexp(quantiles, rates))

    # Example rate parameters (unique)
    rates = [1.0, 2.0, 3.0]

    # Probabilities for which to compute the quantiles
    probabilities = [0.1, 0.5, 0.9]

    print("Quantiles (0.6239176 1.5784264 3.3664883):", quantiles)

    print("Probabilities:", phypoexp(quantiles, rates, lower_tail=False))

    print("Valid:", phypoexp(155.44046020507812, [1.14291730e-03, 1.00100909e-03, 1.00570087e-03, 1.01276805e-03, 3.13469561e-03, 4.17498402e-03, 3.94849390e-01, 1.36364429e-03, 2.40426715e-02, 1.29163908e-03, 1.39841311e+00, 2.50156956e-02, 3.41507978e-03, 5.97681222e-02, 5.81736219e+00, 5.50780718e-02, 2.70972507e-03, 1.23396749e-03, 2.40948416e+00, 6.66804650e-01, 1.01360466e-03, 4.73835500e+00, 3.19146893e-01,
          8.09710967e-02, 5.65116723e-03, 3.63192446e-01, 4.74472783e-03, 3.11148942e+01, 4.00818588e+00, 1.14603525e-02, 1.08729877e+00, 1.01922316e-01, 7.22412252e-01, 8.95276827e-02, 1.00928482e-03, 7.66143124e-03, 6.52676004e-02, 7.75252333e+01, 5.64366043e+01, 1.00102840e-03, 1.20470928e-01, 1.31190146e-01, 1.01019717e-03, 1.15260914e-03, 1.01697769e-03, 9.66367887e-03, 1.48705655e-03, 1.02932046e-03, 1.25623858e+00, 1.85349088e+00]))

    # Example p-value matrix with 3 genes and 4 tests
    pvalues = [
        [0.01, 0.02, 0.03, 0.04],
        [0.05, 0.06, 0.07, 0.08],
        [0.09, 0.10, 0.11, 0.12]
    ]

    # Optionally, define weights (each row should sum to 1)
    weights = [
        [0.25, 0.25, 0.25, 0.25],
        [0.1, 0.2, 0.3, 0.4],
        [0.4, 0.3, 0.2, 0.1]
    ]

    # Combine p-values without specifying weights (equal weights assumed)
    combined_pvalues = cauchy_combine(pvalues)

    print("Combined p-values with equal weights:", combined_pvalues)

    # Combine p-values with specified weights
    combined_pvalues_with_weights = cauchy_combine(pvalues, weights=weights)

    print("Combined p-values with specified weights:", combined_pvalues_with_weights)
