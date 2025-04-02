from torch import rand, randperm, arange


def transform_key_func(generator, n, vocab_size, eff_vocab_size=None):
    # pi = randperm(vocab_size, generator=generator)
    # TODO(doccstat): For the illustration purposes, we are using the
    # trivial permutation here.
    pi = arange(vocab_size)
    xi = rand((n, 1), generator=generator)

    return xi, pi
