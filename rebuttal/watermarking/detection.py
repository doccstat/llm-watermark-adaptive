from torch import unique

from torch import Generator

from numpy import array, sum as np_sum, sum as np_sum
from torch import (arange, argsort, asarray, empty as torch_empty, randint, max
                   as torch_max, min as torch_min)


def permutation_test(tokens,
                     vocab_size,
                     n,
                     k,
                     seed,
                     test_stats,
                     n_runs=100,
                     max_seed=100000):
    generator = Generator()

    test_results = []

    for test_stat in test_stats:
        generator.manual_seed(int(seed))
        test_result = test_stat(tokens=tokens,
                                n=n,
                                k=k,
                                generator=generator,
                                vocab_size=vocab_size)
        test_results.append(test_result)

    test_results = array(test_results)
    null_results = []
    for run in range(n_runs):
        null_results.append([])

        seed = randint(high=max_seed, size=(1,)).item()
        for test_stat in test_stats:
            generator.manual_seed(int(seed))
            null_result = test_stat(tokens=tokens,
                                    n=n,
                                    k=k,
                                    generator=generator,
                                    vocab_size=vocab_size,
                                    null=True)
            null_results[-1].append(null_result)
    null_results = array(null_results)

    return (np_sum(null_results <= test_results, axis=0) + 1.0) / (n_runs + 1.0)


def phi(tokens,
        n,
        k,
        generator,
        key_func,
        vocab_size,
        dist,
        empty_probs,
        ntps,
        null=False,
        normalize=False,
        asis=True):
    if null:
        tokens = unique(asarray(tokens), return_inverse=True, sorted=False)[1]
        eff_vocab_size = torch_max(tokens) + 1
    else:
        eff_vocab_size = vocab_size

    xi, pi = key_func(generator, n, vocab_size, eff_vocab_size)
    tokens = argsort(pi)[tokens]
    # TODO(doccstat): The correctness of the following line is questionable.
    # But for `pi = arange(vocab_size)`, it should not matter.
    # ntps = ntps[argsort(pi), :]

    if normalize:
        tokens = tokens.float() / vocab_size

    A = adjacency(tokens, xi, dist, k, empty_probs, ntps)
    if asis:
        return A
    closest = torch_min(A, axis=1)[0]

    return torch_min(closest)


def adjacency(tokens, xi, dist, k, empty_probs, ntps):
    m = len(tokens)
    n = len(xi)

    A = torch_empty(size=(m - (k - 1), n))
    for i in range(m - (k - 1)):
        for j in range(n):
            A[i][j] = dist(tokens[i:i + k], xi[(j + arange(k)) % n],
                           empty_probs[i:i + k], ntps[:, i:i + k])

    return A
