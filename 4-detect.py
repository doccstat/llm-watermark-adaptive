import time
import torch

from collections import defaultdict
import copy

import numpy as np
from numpy import genfromtxt

from watermarking.detection import permutation_test, phi, quantile_test

from watermarking.transform.score import transform_score
from watermarking.transform.key import transform_key_func

from watermarking.gumbel.score import ems_score, ems_adaptive
from watermarking.gumbel.key import gumbel_key_func

import argparse
import os.path

import csv
import sys

results = defaultdict(dict)

parser = argparse.ArgumentParser(description="Experiment Settings")

parser.add_argument('--method', default="transform", type=str)

parser.add_argument('--model', default="facebook/opt-1.3b", type=str)
parser.add_argument('--token_file', default="", type=str)
parser.add_argument('--seed', default=0, type=int)

parser.add_argument('--k', default=0, type=int)
parser.add_argument('--n', default=256, type=int)
parser.add_argument('--Tindex', default=1, type=int)

parser.add_argument('--prompt_tokens', default=50, type=int)
parser.add_argument('--buffer_tokens', default=20, type=int)
parser.add_argument('--n_runs', default=999, type=int)

parser.add_argument('--gamma', default=0.4, type=float)
parser.add_argument('--truncate_vocab', default=8, type=int)

args = parser.parse_args()
results['args'] = copy.deepcopy(args)

log_file = open(
    'log/' + str(args.Tindex) + "-" +
    args.token_file.split('results/')[1].split('.p')[0] + '.log', 'w'
)
log_file.write(str(args) + '\n')
log_file.flush()

t0 = time.time()

if args.model == "facebook/opt-1.3b":
    vocab_size = 50272
elif args.model == "openai-community/gpt2":
    vocab_size = 50257
elif args.model == "meta-llama/Meta-Llama-3-8B":
    vocab_size = 128256
elif args.model == "mistralai/Mistral-7B-v0.1":
    vocab_size = 32000
else:
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(args.model).to(
        torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
    print(model.get_output_embeddings().weight.shape[0])
    raise
eff_vocab_size = vocab_size - args.truncate_vocab
log_file.write(f'Loaded the model (t = {time.time()-t0} seconds)\n')
log_file.flush()

prompt_tokens = args.prompt_tokens      # minimum prompt length
buffer_tokens = args.buffer_tokens
k = args.k
n = args.n     # watermark key length

seeds = np.genfromtxt(args.token_file + '-seeds.csv',
                      delimiter=',', max_rows=1)

watermarked_samples = genfromtxt(
    args.token_file + '-attacked-tokens.csv', delimiter=",")
# null_samples = genfromtxt(args.token_file + '-null.csv', delimiter=",")
Tindex = min(args.Tindex, watermarked_samples.shape[0])
log_file.write(f'Loaded the samples (t = {time.time()-t0} seconds)\n')
log_file.flush()


if args.method == "transform":
    test_stats = []

    empty_probs = torch.from_numpy(genfromtxt(
        args.token_file + '-re-calculated-empty-probs.csv', delimiter=','
    )[Tindex, :])

    def dist_its(x, y, probs): return transform_score(x, y)

    def test_stat_its(
        tokens, n, k, generator, vocab_size, null=False
    ):
        return phi(
            tokens=tokens,
            n=n,
            k=k,
            generator=generator,
            key_func=transform_key_func,
            vocab_size=vocab_size,
            dist=dist_its,
            empty_probs=empty_probs,
            null=False,
            normalize=True
        )
    test_stats.append(test_stat_its)


elif args.method == "gumbel":
    test_stats = []

    true_probs = torch.from_numpy(genfromtxt(
        args.token_file + '-re-calculated-probs.csv', delimiter=','
    )[Tindex, :])
    empty_probs = torch.from_numpy(genfromtxt(
        args.token_file + '-re-calculated-empty-probs.csv', delimiter=','
    )[Tindex, :])
    best_probs = torch.from_numpy(genfromtxt(
        args.token_file + '-re-calculated-best-probs.csv', delimiter=','
    )[Tindex, :])
    icl_probs = torch.from_numpy(genfromtxt(
        args.token_file + '-re-calculated-icl-probs.csv', delimiter=','
    )[Tindex, :])

    def metric_ems(x, y, probs):
        return ems_score(x, y)

    def test_stat_ems(
        tokens, n, k, generator, vocab_size, null=False
    ):
        return phi(
            tokens=tokens,
            n=n,
            k=k,
            generator=generator,
            key_func=gumbel_key_func,
            vocab_size=vocab_size,
            dist=metric_ems,
            empty_probs=empty_probs,
            null=null,
            normalize=False,
            asis=False
        )
    test_stats.append(test_stat_ems)

    # `true_probs`

    def metric_ems_adaptive_true(x, y, probs):
        return ems_adaptive(x, y, probs, 1.0)

    def test_stat_ems_adaptive_true(
        tokens, n, k, generator, vocab_size, null=False
    ):
        return phi(
            tokens=tokens,
            n=n,
            k=k,
            generator=generator,
            key_func=gumbel_key_func,
            vocab_size=vocab_size,
            dist=metric_ems_adaptive_true,
            empty_probs=true_probs,
            null=null,
            normalize=False,
            asis=False
        )
    test_stats.append(test_stat_ems_adaptive_true)

    # `empty_probs`

    def metric_ems_adaptive_empty_1(x, y, probs):
        return ems_adaptive(x, y, probs, 1.0)

    def test_stat_ems_adaptive_empty_1(
        tokens, n, k, generator, vocab_size, null=False
    ):
        return phi(
            tokens=tokens,
            n=n,
            k=k,
            generator=generator,
            key_func=gumbel_key_func,
            vocab_size=vocab_size,
            dist=metric_ems_adaptive_empty_1,
            empty_probs=empty_probs,
            null=null,
            normalize=False,
            asis=False
        )
    test_stats.append(test_stat_ems_adaptive_empty_1)

    def metric_ems_adaptive_empty_2(x, y, probs):
        return ems_adaptive(x, y, probs, 1.0, 0.0, 0.001)

    def test_stat_ems_adaptive_empty_2(
        tokens, n, k, generator, vocab_size, null=False
    ):
        return phi(
            tokens=tokens,
            n=n,
            k=k,
            generator=generator,
            key_func=gumbel_key_func,
            vocab_size=vocab_size,
            dist=metric_ems_adaptive_empty_2,
            empty_probs=empty_probs,
            null=null,
            normalize=False,
            asis=False
        )
    test_stats.append(test_stat_ems_adaptive_empty_2)

    def metric_ems_adaptive_empty_3(x, y, probs):
        return ems_adaptive(x, y, probs, 1.0, 0.0, 0.01)

    def test_stat_ems_adaptive_empty_3(
        tokens, n, k, generator, vocab_size, null=False
    ):
        return phi(
            tokens=tokens,
            n=n,
            k=k,
            generator=generator,
            key_func=gumbel_key_func,
            vocab_size=vocab_size,
            dist=metric_ems_adaptive_empty_3,
            empty_probs=empty_probs,
            null=null,
            normalize=False,
            asis=False
        )
    test_stats.append(test_stat_ems_adaptive_empty_3)

    def metric_ems_adaptive_empty_4(x, y, probs):
        return ems_adaptive(x, y, probs, 1.0, 0.0, 0.1)

    def test_stat_ems_adaptive_empty_4(
        tokens, n, k, generator, vocab_size, null=False
    ):
        return phi(
            tokens=tokens,
            n=n,
            k=k,
            generator=generator,
            key_func=gumbel_key_func,
            vocab_size=vocab_size,
            dist=metric_ems_adaptive_empty_4,
            empty_probs=empty_probs,
            null=null,
            normalize=False,
            asis=False
        )
    test_stats.append(test_stat_ems_adaptive_empty_4)

    def metric_ems_adaptive_empty_5(x, y, probs):
        return ems_adaptive(x, y, probs, 1.0, 0.0, 0.0, 0.9)

    def test_stat_ems_adaptive_empty_5(
        tokens, n, k, generator, vocab_size, null=False
    ):
        return phi(
            tokens=tokens,
            n=n,
            k=k,
            generator=generator,
            key_func=gumbel_key_func,
            vocab_size=vocab_size,
            dist=metric_ems_adaptive_empty_5,
            empty_probs=empty_probs,
            null=null,
            normalize=False,
            asis=False
        )
    test_stats.append(test_stat_ems_adaptive_empty_5)

    def metric_ems_adaptive_empty_6(x, y, probs):
        return ems_adaptive(x, y, probs, 1.0, 0.0, 0.0, 0.8)

    def test_stat_ems_adaptive_empty_6(
        tokens, n, k, generator, vocab_size, null=False
    ):
        return phi(
            tokens=tokens,
            n=n,
            k=k,
            generator=generator,
            key_func=gumbel_key_func,
            vocab_size=vocab_size,
            dist=metric_ems_adaptive_empty_6,
            empty_probs=empty_probs,
            null=null,
            normalize=False,
            asis=False
        )
    test_stats.append(test_stat_ems_adaptive_empty_6)

    def metric_ems_adaptive_empty_7(x, y, probs):
        return ems_adaptive(x, y, probs, 1.0, 0.0, 0.0, 0.7)

    def test_stat_ems_adaptive_empty_7(
        tokens, n, k, generator, vocab_size, null=False
    ):
        return phi(
            tokens=tokens,
            n=n,
            k=k,
            generator=generator,
            key_func=gumbel_key_func,
            vocab_size=vocab_size,
            dist=metric_ems_adaptive_empty_7,
            empty_probs=empty_probs,
            null=null,
            normalize=False,
            asis=False
        )
    test_stats.append(test_stat_ems_adaptive_empty_7)

    def metric_ems_adaptive_empty_8(x, y, probs):
        return ems_adaptive(x, y, probs, 1.0, 0.0, 0.0, 0.6)

    def test_stat_ems_adaptive_empty_8(
        tokens, n, k, generator, vocab_size, null=False
    ):
        return phi(
            tokens=tokens,
            n=n,
            k=k,
            generator=generator,
            key_func=gumbel_key_func,
            vocab_size=vocab_size,
            dist=metric_ems_adaptive_empty_8,
            empty_probs=empty_probs,
            null=null,
            normalize=False,
            asis=False
        )
    test_stats.append(test_stat_ems_adaptive_empty_8)

    def metric_ems_adaptive_empty_9(x, y, probs):
        return ems_adaptive(x, y, probs, 1.0, 0.0, 0.0, 0.5)

    def test_stat_ems_adaptive_empty_9(
        tokens, n, k, generator, vocab_size, null=False
    ):
        return phi(
            tokens=tokens,
            n=n,
            k=k,
            generator=generator,
            key_func=gumbel_key_func,
            vocab_size=vocab_size,
            dist=metric_ems_adaptive_empty_9,
            empty_probs=empty_probs,
            null=null,
            normalize=False,
            asis=False
        )
    test_stats.append(test_stat_ems_adaptive_empty_9)

    def metric_ems_adaptive_empty_10(x, y, probs):
        return ems_adaptive(x, y, probs, 1.0, 0.0, 0.0, 0.4)

    def test_stat_ems_adaptive_empty_10(
        tokens, n, k, generator, vocab_size, null=False
    ):
        return phi(
            tokens=tokens,
            n=n,
            k=k,
            generator=generator,
            key_func=gumbel_key_func,
            vocab_size=vocab_size,
            dist=metric_ems_adaptive_empty_10,
            empty_probs=empty_probs,
            null=null,
            normalize=False,
            asis=False
        )
    test_stats.append(test_stat_ems_adaptive_empty_10)

    def metric_ems_adaptive_empty_11(x, y, probs):
        return ems_adaptive(x, y, probs, 1.0, 0.0, 0.0, 0.3)

    def test_stat_ems_adaptive_empty_11(
        tokens, n, k, generator, vocab_size, null=False
    ):
        return phi(
            tokens=tokens,
            n=n,
            k=k,
            generator=generator,
            key_func=gumbel_key_func,
            vocab_size=vocab_size,
            dist=metric_ems_adaptive_empty_11,
            empty_probs=empty_probs,
            null=null,
            normalize=False,
            asis=False
        )
    test_stats.append(test_stat_ems_adaptive_empty_11)

    def metric_ems_adaptive_empty_12(x, y, probs):
        return ems_adaptive(x, y, probs, 1.0, 0.0, 0.0, 0.2)

    def test_stat_ems_adaptive_empty_12(
        tokens, n, k, generator, vocab_size, null=False
    ):
        return phi(
            tokens=tokens,
            n=n,
            k=k,
            generator=generator,
            key_func=gumbel_key_func,
            vocab_size=vocab_size,
            dist=metric_ems_adaptive_empty_12,
            empty_probs=empty_probs,
            null=null,
            normalize=False,
            asis=False
        )
    test_stats.append(test_stat_ems_adaptive_empty_12)

    def metric_ems_adaptive_empty_13(x, y, probs):
        return ems_adaptive(x, y, probs, 1.0, 0.0, 0.0, 0.1)

    def test_stat_ems_adaptive_empty_13(
        tokens, n, k, generator, vocab_size, null=False
    ):
        return phi(
            tokens=tokens,
            n=n,
            k=k,
            generator=generator,
            key_func=gumbel_key_func,
            vocab_size=vocab_size,
            dist=metric_ems_adaptive_empty_13,
            empty_probs=empty_probs,
            null=null,
            normalize=False,
            asis=False
        )
    test_stats.append(test_stat_ems_adaptive_empty_13)

    # `best_probs`

    def metric_ems_adaptive_best_1(x, y, probs):
        return ems_adaptive(x, y, probs, 1.0)

    def test_stat_ems_adaptive_best_1(
        tokens, n, k, generator, vocab_size, null=False
    ):
        return phi(
            tokens=tokens,
            n=n,
            k=k,
            generator=generator,
            key_func=gumbel_key_func,
            vocab_size=vocab_size,
            dist=metric_ems_adaptive_best_1,
            empty_probs=best_probs,
            null=null,
            normalize=False,
            asis=False
        )
    test_stats.append(test_stat_ems_adaptive_best_1)

    def metric_ems_adaptive_best_2(x, y, probs):
        return ems_adaptive(x, y, probs, 1.0, 0.0, 0.001)

    def test_stat_ems_adaptive_best_2(
        tokens, n, k, generator, vocab_size, null=False
    ):
        return phi(
            tokens=tokens,
            n=n,
            k=k,
            generator=generator,
            key_func=gumbel_key_func,
            vocab_size=vocab_size,
            dist=metric_ems_adaptive_best_2,
            empty_probs=best_probs,
            null=null,
            normalize=False,
            asis=False
        )
    test_stats.append(test_stat_ems_adaptive_best_2)

    def metric_ems_adaptive_best_3(x, y, probs):
        return ems_adaptive(x, y, probs, 1.0, 0.0, 0.01)

    def test_stat_ems_adaptive_best_3(
        tokens, n, k, generator, vocab_size, null=False
    ):
        return phi(
            tokens=tokens,
            n=n,
            k=k,
            generator=generator,
            key_func=gumbel_key_func,
            vocab_size=vocab_size,
            dist=metric_ems_adaptive_best_3,
            empty_probs=best_probs,
            null=null,
            normalize=False,
            asis=False
        )
    test_stats.append(test_stat_ems_adaptive_best_3)

    def metric_ems_adaptive_best_4(x, y, probs):
        return ems_adaptive(x, y, probs, 1.0, 0.0, 0.1)

    def test_stat_ems_adaptive_best_4(
        tokens, n, k, generator, vocab_size, null=False
    ):
        return phi(
            tokens=tokens,
            n=n,
            k=k,
            generator=generator,
            key_func=gumbel_key_func,
            vocab_size=vocab_size,
            dist=metric_ems_adaptive_best_4,
            empty_probs=best_probs,
            null=null,
            normalize=False,
            asis=False
        )
    test_stats.append(test_stat_ems_adaptive_best_4)

    def metric_ems_adaptive_best_5(x, y, probs):
        return ems_adaptive(x, y, probs, 1.0, 0.0, 0.0, 0.9)

    def test_stat_ems_adaptive_best_5(
        tokens, n, k, generator, vocab_size, null=False
    ):
        return phi(
            tokens=tokens,
            n=n,
            k=k,
            generator=generator,
            key_func=gumbel_key_func,
            vocab_size=vocab_size,
            dist=metric_ems_adaptive_best_5,
            empty_probs=best_probs,
            null=null,
            normalize=False,
            asis=False
        )
    test_stats.append(test_stat_ems_adaptive_best_5)

    def metric_ems_adaptive_best_6(x, y, probs):
        return ems_adaptive(x, y, probs, 1.0, 0.0, 0.0, 0.8)

    def test_stat_ems_adaptive_best_6(
        tokens, n, k, generator, vocab_size, null=False
    ):
        return phi(
            tokens=tokens,
            n=n,
            k=k,
            generator=generator,
            key_func=gumbel_key_func,
            vocab_size=vocab_size,
            dist=metric_ems_adaptive_best_6,
            empty_probs=best_probs,
            null=null,
            normalize=False,
            asis=False
        )
    test_stats.append(test_stat_ems_adaptive_best_6)

    def metric_ems_adaptive_best_7(x, y, probs):
        return ems_adaptive(x, y, probs, 1.0, 0.0, 0.0, 0.7)

    def test_stat_ems_adaptive_best_7(
        tokens, n, k, generator, vocab_size, null=False
    ):
        return phi(
            tokens=tokens,
            n=n,
            k=k,
            generator=generator,
            key_func=gumbel_key_func,
            vocab_size=vocab_size,
            dist=metric_ems_adaptive_best_7,
            empty_probs=best_probs,
            null=null,
            normalize=False,
            asis=False
        )
    test_stats.append(test_stat_ems_adaptive_best_7)

    def metric_ems_adaptive_best_8(x, y, probs):
        return ems_adaptive(x, y, probs, 1.0, 0.0, 0.0, 0.6)

    def test_stat_ems_adaptive_best_8(
        tokens, n, k, generator, vocab_size, null=False
    ):
        return phi(
            tokens=tokens,
            n=n,
            k=k,
            generator=generator,
            key_func=gumbel_key_func,
            vocab_size=vocab_size,
            dist=metric_ems_adaptive_best_8,
            empty_probs=best_probs,
            null=null,
            normalize=False,
            asis=False
        )
    test_stats.append(test_stat_ems_adaptive_best_8)

    def metric_ems_adaptive_best_9(x, y, probs):
        return ems_adaptive(x, y, probs, 1.0, 0.0, 0.0, 0.5)

    def test_stat_ems_adaptive_best_9(
        tokens, n, k, generator, vocab_size, null=False
    ):
        return phi(
            tokens=tokens,
            n=n,
            k=k,
            generator=generator,
            key_func=gumbel_key_func,
            vocab_size=vocab_size,
            dist=metric_ems_adaptive_best_9,
            empty_probs=best_probs,
            null=null,
            normalize=False,
            asis=False
        )
    test_stats.append(test_stat_ems_adaptive_best_9)

    def metric_ems_adaptive_best_10(x, y, probs):
        return ems_adaptive(x, y, probs, 1.0, 0.0, 0.0, 0.4)

    def test_stat_ems_adaptive_best_10(
        tokens, n, k, generator, vocab_size, null=False
    ):
        return phi(
            tokens=tokens,
            n=n,
            k=k,
            generator=generator,
            key_func=gumbel_key_func,
            vocab_size=vocab_size,
            dist=metric_ems_adaptive_best_10,
            empty_probs=best_probs,
            null=null,
            normalize=False,
            asis=False
        )
    test_stats.append(test_stat_ems_adaptive_best_10)

    def metric_ems_adaptive_best_11(x, y, probs):
        return ems_adaptive(x, y, probs, 1.0, 0.0, 0.0, 0.3)

    def test_stat_ems_adaptive_best_11(
        tokens, n, k, generator, vocab_size, null=False
    ):
        return phi(
            tokens=tokens,
            n=n,
            k=k,
            generator=generator,
            key_func=gumbel_key_func,
            vocab_size=vocab_size,
            dist=metric_ems_adaptive_best_11,
            empty_probs=best_probs,
            null=null,
            normalize=False,
            asis=False
        )
    test_stats.append(test_stat_ems_adaptive_best_11)

    def metric_ems_adaptive_best_12(x, y, probs):
        return ems_adaptive(x, y, probs, 1.0, 0.0, 0.0, 0.2)

    def test_stat_ems_adaptive_best_12(
        tokens, n, k, generator, vocab_size, null=False
    ):
        return phi(
            tokens=tokens,
            n=n,
            k=k,
            generator=generator,
            key_func=gumbel_key_func,
            vocab_size=vocab_size,
            dist=metric_ems_adaptive_best_12,
            empty_probs=best_probs,
            null=null,
            normalize=False,
            asis=False
        )
    test_stats.append(test_stat_ems_adaptive_best_12)

    def metric_ems_adaptive_best_13(x, y, probs):
        return ems_adaptive(x, y, probs, 1.0, 0.0, 0.0, 0.1)

    def test_stat_ems_adaptive_best_13(
        tokens, n, k, generator, vocab_size, null=False
    ):
        return phi(
            tokens=tokens,
            n=n,
            k=k,
            generator=generator,
            key_func=gumbel_key_func,
            vocab_size=vocab_size,
            dist=metric_ems_adaptive_best_13,
            empty_probs=best_probs,
            null=null,
            normalize=False,
            asis=False
        )
    test_stats.append(test_stat_ems_adaptive_best_13)

    # `icl_probs`

    def metric_ems_adaptive_icl_1(x, y, probs):
        return ems_adaptive(x, y, probs, 1.0)

    def test_stat_ems_adaptive_icl_1(
        tokens, n, k, generator, vocab_size, null=False
    ):
        return phi(
            tokens=tokens,
            n=n,
            k=k,
            generator=generator,
            key_func=gumbel_key_func,
            vocab_size=vocab_size,
            dist=metric_ems_adaptive_icl_1,
            empty_probs=icl_probs,
            null=null,
            normalize=False,
            asis=False
        )
    test_stats.append(test_stat_ems_adaptive_icl_1)

    def metric_ems_adaptive_icl_2(x, y, probs):
        return ems_adaptive(x, y, probs, 1.0, 0.0, 0.001)

    def test_stat_ems_adaptive_icl_2(
        tokens, n, k, generator, vocab_size, null=False
    ):
        return phi(
            tokens=tokens,
            n=n,
            k=k,
            generator=generator,
            key_func=gumbel_key_func,
            vocab_size=vocab_size,
            dist=metric_ems_adaptive_icl_2,
            empty_probs=icl_probs,
            null=null,
            normalize=False,
            asis=False
        )
    test_stats.append(test_stat_ems_adaptive_icl_2)

    def metric_ems_adaptive_icl_3(x, y, probs):
        return ems_adaptive(x, y, probs, 1.0, 0.0, 0.01)

    def test_stat_ems_adaptive_icl_3(
        tokens, n, k, generator, vocab_size, null=False
    ):
        return phi(
            tokens=tokens,
            n=n,
            k=k,
            generator=generator,
            key_func=gumbel_key_func,
            vocab_size=vocab_size,
            dist=metric_ems_adaptive_icl_3,
            empty_probs=icl_probs,
            null=null,
            normalize=False,
            asis=False
        )
    test_stats.append(test_stat_ems_adaptive_icl_3)

    def metric_ems_adaptive_icl_4(x, y, probs):
        return ems_adaptive(x, y, probs, 1.0, 0.0, 0.1)

    def test_stat_ems_adaptive_icl_4(
        tokens, n, k, generator, vocab_size, null=False
    ):
        return phi(
            tokens=tokens,
            n=n,
            k=k,
            generator=generator,
            key_func=gumbel_key_func,
            vocab_size=vocab_size,
            dist=metric_ems_adaptive_icl_4,
            empty_probs=icl_probs,
            null=null,
            normalize=False,
            asis=False
        )
    test_stats.append(test_stat_ems_adaptive_icl_4)

    def metric_ems_adaptive_icl_5(x, y, probs):
        return ems_adaptive(x, y, probs, 1.0, 0.0, 0.0, 0.9)

    def test_stat_ems_adaptive_icl_5(
        tokens, n, k, generator, vocab_size, null=False
    ):
        return phi(
            tokens=tokens,
            n=n,
            k=k,
            generator=generator,
            key_func=gumbel_key_func,
            vocab_size=vocab_size,
            dist=metric_ems_adaptive_icl_5,
            empty_probs=icl_probs,
            null=null,
            normalize=False,
            asis=False
        )
    test_stats.append(test_stat_ems_adaptive_icl_5)

    def metric_ems_adaptive_icl_6(x, y, probs):
        return ems_adaptive(x, y, probs, 1.0, 0.0, 0.0, 0.8)

    def test_stat_ems_adaptive_icl_6(
        tokens, n, k, generator, vocab_size, null=False
    ):
        return phi(
            tokens=tokens,
            n=n,
            k=k,
            generator=generator,
            key_func=gumbel_key_func,
            vocab_size=vocab_size,
            dist=metric_ems_adaptive_icl_6,
            empty_probs=icl_probs,
            null=null,
            normalize=False,
            asis=False
        )
    test_stats.append(test_stat_ems_adaptive_icl_6)

    def metric_ems_adaptive_icl_7(x, y, probs):
        return ems_adaptive(x, y, probs, 1.0, 0.0, 0.0, 0.7)

    def test_stat_ems_adaptive_icl_7(
        tokens, n, k, generator, vocab_size, null=False
    ):
        return phi(
            tokens=tokens,
            n=n,
            k=k,
            generator=generator,
            key_func=gumbel_key_func,
            vocab_size=vocab_size,
            dist=metric_ems_adaptive_icl_7,
            empty_probs=icl_probs,
            null=null,
            normalize=False,
            asis=False
        )
    test_stats.append(test_stat_ems_adaptive_icl_7)

    def metric_ems_adaptive_icl_8(x, y, probs):
        return ems_adaptive(x, y, probs, 1.0, 0.0, 0.0, 0.6)

    def test_stat_ems_adaptive_icl_8(
        tokens, n, k, generator, vocab_size, null=False
    ):
        return phi(
            tokens=tokens,
            n=n,
            k=k,
            generator=generator,
            key_func=gumbel_key_func,
            vocab_size=vocab_size,
            dist=metric_ems_adaptive_icl_8,
            empty_probs=icl_probs,
            null=null,
            normalize=False,
            asis=False
        )
    test_stats.append(test_stat_ems_adaptive_icl_8)

    def metric_ems_adaptive_icl_9(x, y, probs):
        return ems_adaptive(x, y, probs, 1.0, 0.0, 0.0, 0.5)

    def test_stat_ems_adaptive_icl_9(
        tokens, n, k, generator, vocab_size, null=False
    ):
        return phi(
            tokens=tokens,
            n=n,
            k=k,
            generator=generator,
            key_func=gumbel_key_func,
            vocab_size=vocab_size,
            dist=metric_ems_adaptive_icl_9,
            empty_probs=icl_probs,
            null=null,
            normalize=False,
            asis=False
        )
    test_stats.append(test_stat_ems_adaptive_icl_9)

    def metric_ems_adaptive_icl_10(x, y, probs):
        return ems_adaptive(x, y, probs, 1.0, 0.0, 0.0, 0.4)

    def test_stat_ems_adaptive_icl_10(
        tokens, n, k, generator, vocab_size, null=False
    ):
        return phi(
            tokens=tokens,
            n=n,
            k=k,
            generator=generator,
            key_func=gumbel_key_func,
            vocab_size=vocab_size,
            dist=metric_ems_adaptive_icl_10,
            empty_probs=icl_probs,
            null=null,
            normalize=False,
            asis=False
        )
    test_stats.append(test_stat_ems_adaptive_icl_10)

    def metric_ems_adaptive_icl_11(x, y, probs):
        return ems_adaptive(x, y, probs, 1.0, 0.0, 0.0, 0.3)

    def test_stat_ems_adaptive_icl_11(
        tokens, n, k, generator, vocab_size, null=False
    ):
        return phi(
            tokens=tokens,
            n=n,
            k=k,
            generator=generator,
            key_func=gumbel_key_func,
            vocab_size=vocab_size,
            dist=metric_ems_adaptive_icl_11,
            empty_probs=icl_probs,
            null=null,
            normalize=False,
            asis=False
        )
    test_stats.append(test_stat_ems_adaptive_icl_11)

    def metric_ems_adaptive_icl_12(x, y, probs):
        return ems_adaptive(x, y, probs, 1.0, 0.0, 0.0, 0.2)

    def test_stat_ems_adaptive_icl_12(
        tokens, n, k, generator, vocab_size, null=False
    ):
        return phi(
            tokens=tokens,
            n=n,
            k=k,
            generator=generator,
            key_func=gumbel_key_func,
            vocab_size=vocab_size,
            dist=metric_ems_adaptive_icl_12,
            empty_probs=icl_probs,
            null=null,
            normalize=False,
            asis=False
        )
    test_stats.append(test_stat_ems_adaptive_icl_12)

    def metric_ems_adaptive_icl_13(x, y, probs):
        return ems_adaptive(x, y, probs, 1.0, 0.0, 0.0, 0.1)

    def test_stat_ems_adaptive_icl_13(
        tokens, n, k, generator, vocab_size, null=False
    ):
        return phi(
            tokens=tokens,
            n=n,
            k=k,
            generator=generator,
            key_func=gumbel_key_func,
            vocab_size=vocab_size,
            dist=metric_ems_adaptive_icl_13,
            empty_probs=icl_probs,
            null=null,
            normalize=False,
            asis=False
        )
    test_stats.append(test_stat_ems_adaptive_icl_13)

    # Probabilities calculated with 20% prompt (80% modified)
    probs_20 = genfromtxt(
        args.token_file + '-re-calculated-20-probs.csv', delimiter=','
    )[Tindex, :]
    probs_20 = torch.from_numpy(probs_20)

    def metric_ems_adaptive_probs_20(x, y, probs):
        return ems_adaptive(x, y, probs, 1.0, 0.0, 0.0, 0.5)

    def test_stat_ems_adaptive_probs_20(
        tokens, n, k, generator, vocab_size, null=False
    ):
        return phi(
            tokens=tokens,
            n=n,
            k=k,
            generator=generator,
            key_func=gumbel_key_func,
            vocab_size=vocab_size,
            dist=metric_ems_adaptive_probs_20,
            empty_probs=probs_20,
            null=null,
            normalize=False,
            asis=False
        )
    test_stats.append(test_stat_ems_adaptive_probs_20)

    # Probabilities calculated with 40% prompt (60% modified)
    probs_40 = genfromtxt(
        args.token_file + '-re-calculated-40-probs.csv', delimiter=','
    )[Tindex, :]
    probs_40 = torch.from_numpy(probs_40)

    def metric_ems_adaptive_probs_40(x, y, probs):
        return ems_adaptive(x, y, probs, 1.0, 0.0, 0.0, 0.5)

    def test_stat_ems_adaptive_probs_40(
        tokens, n, k, generator, vocab_size, null=False
    ):
        return phi(
            tokens=tokens,
            n=n,
            k=k,
            generator=generator,
            key_func=gumbel_key_func,
            vocab_size=vocab_size,
            dist=metric_ems_adaptive_probs_40,
            empty_probs=probs_40,
            null=null,
            normalize=False,
            asis=False
        )
    test_stats.append(test_stat_ems_adaptive_probs_40)

    # Probabilities calculated with 60% prompt (40% modified)
    probs_60 = genfromtxt(
        args.token_file + '-re-calculated-60-probs.csv', delimiter=','
    )[Tindex, :]
    probs_60 = torch.from_numpy(probs_60)

    def metric_ems_adaptive_probs_60(x, y, probs):
        return ems_adaptive(x, y, probs, 1.0, 0.0, 0.0, 0.5)

    def test_stat_ems_adaptive_probs_60(
        tokens, n, k, generator, vocab_size, null=False
    ):
        return phi(
            tokens=tokens,
            n=n,
            k=k,
            generator=generator,
            key_func=gumbel_key_func,
            vocab_size=vocab_size,
            dist=metric_ems_adaptive_probs_60,
            empty_probs=probs_60,
            null=null,
            normalize=False,
            asis=False
        )
    test_stats.append(test_stat_ems_adaptive_probs_60)

    # Probabilities calculated with 80% prompt (20% modified)
    probs_80 = genfromtxt(
        args.token_file + '-re-calculated-80-probs.csv', delimiter=','
    )[Tindex, :]
    probs_80 = torch.from_numpy(probs_80)

    def metric_ems_adaptive_probs_80(x, y, probs):
        return ems_adaptive(x, y, probs, 1.0, 0.0, 0.0, 0.5)

    def test_stat_ems_adaptive_probs_80(
        tokens, n, k, generator, vocab_size, null=False
    ):
        return phi(
            tokens=tokens,
            n=n,
            k=k,
            generator=generator,
            key_func=gumbel_key_func,
            vocab_size=vocab_size,
            dist=metric_ems_adaptive_probs_80,
            empty_probs=probs_80,
            null=null,
            normalize=False,
            asis=False
        )
    test_stats.append(test_stat_ems_adaptive_probs_80)

    # Probabilities calculated with 90% prompt (10% modified)
    probs_90 = genfromtxt(
        args.token_file + '-re-calculated-90-probs.csv', delimiter=','
    )[Tindex, :]
    probs_90 = torch.from_numpy(probs_90)

    def metric_ems_adaptive_probs_90(x, y, probs):
        return ems_adaptive(x, y, probs, 1.0, 0.0, 0.0, 0.5)

    def test_stat_ems_adaptive_probs_90(
        tokens, n, k, generator, vocab_size, null=False
    ):
        return phi(
            tokens=tokens,
            n=n,
            k=k,
            generator=generator,
            key_func=gumbel_key_func,
            vocab_size=vocab_size,
            dist=metric_ems_adaptive_probs_90,
            empty_probs=probs_90,
            null=null,
            normalize=False,
            asis=False
        )
    test_stats.append(test_stat_ems_adaptive_probs_90)

    # Probabilities calculated with 96% prompt (4% modified)
    probs_96 = genfromtxt(
        args.token_file + '-re-calculated-96-probs.csv', delimiter=','
    )[Tindex, :]
    probs_96 = torch.from_numpy(probs_96)

    def metric_ems_adaptive_probs_96(x, y, probs):
        return ems_adaptive(x, y, probs, 1.0, 0.0, 0.0, 0.5)

    def test_stat_ems_adaptive_probs_96(
        tokens, n, k, generator, vocab_size, null=False
    ):
        return phi(
            tokens=tokens,
            n=n,
            k=k,
            generator=generator,
            key_func=gumbel_key_func,
            vocab_size=vocab_size,
            dist=metric_ems_adaptive_probs_96,
            empty_probs=probs_96,
            null=null,
            normalize=False,
            asis=False
        )
    test_stats.append(test_stat_ems_adaptive_probs_96)

    # Probabilities calculated with 98% prompt (2% modified)
    probs_98 = genfromtxt(
        args.token_file + '-re-calculated-98-probs.csv', delimiter=','
    )[Tindex, :]
    probs_98 = torch.from_numpy(probs_98)

    def metric_ems_adaptive_probs_98(x, y, probs):
        return ems_adaptive(x, y, probs, 1.0, 0.0, 0.0, 0.5)

    def test_stat_ems_adaptive_probs_98(
        tokens, n, k, generator, vocab_size, null=False
    ):
        return phi(
            tokens=tokens,
            n=n,
            k=k,
            generator=generator,
            key_func=gumbel_key_func,
            vocab_size=vocab_size,
            dist=metric_ems_adaptive_probs_98,
            empty_probs=probs_98,
            null=null,
            normalize=False,
            asis=False
        )
    test_stats.append(test_stat_ems_adaptive_probs_98)

    # Probabilities calculated with 20% prompt (80% modified), only calculated for 80% of the tokens
    probs_20_08 = genfromtxt(
        args.token_file + '-re-calculated-20-probs.csv', delimiter=','
    )[Tindex, :]
    probs_20_08 = torch.from_numpy(probs_20_08)

    def metric_ems_adaptive_probs_20_08(x, y, probs):
        return ems_adaptive(x, y, probs, 0.8, 0.0, 0.0, 0.5)

    def test_stat_ems_adaptive_probs_20_08(
        tokens, n, k, generator, vocab_size, null=False
    ):
        return phi(
            tokens=tokens,
            n=n,
            k=k,
            generator=generator,
            key_func=gumbel_key_func,
            vocab_size=vocab_size,
            dist=metric_ems_adaptive_probs_20_08,
            empty_probs=probs_20_08,
            null=null,
            normalize=False,
            asis=False
        )
    test_stats.append(test_stat_ems_adaptive_probs_20_08)

    # Probabilities calculated with 40% prompt (60% modified), only calculated for 80% of the tokens
    probs_40_08 = genfromtxt(
        args.token_file + '-re-calculated-40-probs.csv', delimiter=','
    )[Tindex, :]
    probs_40_08 = torch.from_numpy(probs_40_08)

    def metric_ems_adaptive_probs_40_08(x, y, probs):
        return ems_adaptive(x, y, probs, 0.8, 0.0, 0.0, 0.5)

    def test_stat_ems_adaptive_probs_40_08(
        tokens, n, k, generator, vocab_size, null=False
    ):
        return phi(
            tokens=tokens,
            n=n,
            k=k,
            generator=generator,
            key_func=gumbel_key_func,
            vocab_size=vocab_size,
            dist=metric_ems_adaptive_probs_40_08,
            empty_probs=probs_40_08,
            null=null,
            normalize=False,
            asis=False
        )
    test_stats.append(test_stat_ems_adaptive_probs_40_08)

    # Probabilities calculated with 60% prompt (40% modified), only calculated for 80% of the tokens
    probs_60_08 = genfromtxt(
        args.token_file + '-re-calculated-60-probs.csv', delimiter=','
    )[Tindex, :]
    probs_60_08 = torch.from_numpy(probs_60_08)

    def metric_ems_adaptive_probs_60_08(x, y, probs):
        return ems_adaptive(x, y, probs, 0.8, 0.0, 0.0, 0.5)

    def test_stat_ems_adaptive_probs_60_08(
        tokens, n, k, generator, vocab_size, null=False
    ):
        return phi(
            tokens=tokens,
            n=n,
            k=k,
            generator=generator,
            key_func=gumbel_key_func,
            vocab_size=vocab_size,
            dist=metric_ems_adaptive_probs_60_08,
            empty_probs=probs_60_08,
            null=null,
            normalize=False,
            asis=False
        )
    test_stats.append(test_stat_ems_adaptive_probs_60_08)

    # Probabilities calculated with 80% prompt (20% modified), only calculated for 80% of the tokens
    probs_80_08 = genfromtxt(
        args.token_file + '-re-calculated-80-probs.csv', delimiter=','
    )[Tindex, :]
    probs_80_08 = torch.from_numpy(probs_80_08)

    def metric_ems_adaptive_probs_80_08(x, y, probs):
        return ems_adaptive(x, y, probs, 0.8, 0.0, 0.0, 0.5)

    def test_stat_ems_adaptive_probs_80_08(
        tokens, n, k, generator, vocab_size, null=False
    ):
        return phi(
            tokens=tokens,
            n=n,
            k=k,
            generator=generator,
            key_func=gumbel_key_func,
            vocab_size=vocab_size,
            dist=metric_ems_adaptive_probs_80_08,
            empty_probs=probs_80_08,
            null=null,
            normalize=False,
            asis=False
        )
    test_stats.append(test_stat_ems_adaptive_probs_80_08)

    # Probabilities calculated with 90% prompt (10% modified), only calculated for 80% of the tokens
    probs_90_08 = genfromtxt(
        args.token_file + '-re-calculated-90-probs.csv', delimiter=','
    )[Tindex, :]
    probs_90_08 = torch.from_numpy(probs_90_08)

    def metric_ems_adaptive_probs_90_08(x, y, probs):
        return ems_adaptive(x, y, probs, 0.8, 0.0, 0.0, 0.5)

    def test_stat_ems_adaptive_probs_90_08(
        tokens, n, k, generator, vocab_size, null=False
    ):
        return phi(
            tokens=tokens,
            n=n,
            k=k,
            generator=generator,
            key_func=gumbel_key_func,
            vocab_size=vocab_size,
            dist=metric_ems_adaptive_probs_90_08,
            empty_probs=probs_90_08,
            null=null,
            normalize=False,
            asis=False
        )
    test_stats.append(test_stat_ems_adaptive_probs_90_08)

    # Probabilities calculated with 96% prompt (4% modified), only calculated for 80% of the tokens
    probs_96_08 = genfromtxt(
        args.token_file + '-re-calculated-96-probs.csv', delimiter=','
    )[Tindex, :]
    probs_96_08 = torch.from_numpy(probs_96_08)

    def metric_ems_adaptive_probs_96_08(x, y, probs):
        return ems_adaptive(x, y, probs, 0.8, 0.0, 0.0, 0.5)

    def test_stat_ems_adaptive_probs_96_08(
        tokens, n, k, generator, vocab_size, null=False
    ):
        return phi(
            tokens=tokens,
            n=n,
            k=k,
            generator=generator,
            key_func=gumbel_key_func,
            vocab_size=vocab_size,
            dist=metric_ems_adaptive_probs_96_08,
            empty_probs=probs_96_08,
            null=null,
            normalize=False,
            asis=False
        )
    test_stats.append(test_stat_ems_adaptive_probs_96_08)

    # Probabilities calculated with 98% prompt (2% modified), only calculated for 80% of the tokens
    probs_98_08 = genfromtxt(
        args.token_file + '-re-calculated-98-probs.csv', delimiter=','
    )[Tindex, :]
    probs_98_08 = torch.from_numpy(probs_98_08)

    def metric_ems_adaptive_probs_98_08(x, y, probs):
        return ems_adaptive(x, y, probs, 0.8, 0.0, 0.0, 0.5)

    def test_stat_ems_adaptive_probs_98_08(
        tokens, n, k, generator, vocab_size, null=False
    ):
        return phi(
            tokens=tokens,
            n=n,
            k=k,
            generator=generator,
            key_func=gumbel_key_func,
            vocab_size=vocab_size,
            dist=metric_ems_adaptive_probs_98_08,
            empty_probs=probs_98_08,
            null=null,
            normalize=False,
            asis=False
        )
    test_stats.append(test_stat_ems_adaptive_probs_98_08)

else:
    raise

# Don't forget to remove the folder following the readme file,
# if the experiment needs re-running.
if os.path.exists(f"{args.token_file}-{args.k}-detect/watermarked-{args.Tindex}.csv"):
    with open(
        f"{args.token_file}-{args.k}-detect/watermarked-{args.Tindex}.csv", 'r'
    ) as f:
        reader = csv.reader(f)
        first_row = next(reader, None)
        if first_row is not None and len(first_row) == len(test_stats):
            sys.exit()


def test(tokens, seed, test_stats):
    return permutation_test(tokens,
                            vocab_size,
                            n,
                            k,
                            seed,
                            test_stats,
                            log_file=log_file,
                            n_runs=args.n_runs)
    return quantile_test(
        tokens,
        vocab_size,
        n,
        k,
        seed,
        test_stats,
        [
            torch.from_numpy(np.ones(tokens.shape[0]) * 0.5),
            torch.from_numpy(genfromtxt(
                args.token_file + '-re-calculated-probs.csv', delimiter=','
            )[Tindex, :]),
            torch.from_numpy(genfromtxt(
                args.token_file + '-re-calculated-empty-probs.csv', delimiter=','
            )[Tindex, :]),
            torch.from_numpy(genfromtxt(
                args.token_file + '-re-calculated-best-probs.csv', delimiter=','
            )[Tindex, :]),
            torch.from_numpy(genfromtxt(
                args.token_file + '-re-calculated-icl-probs.csv', delimiter=','
            )[Tindex, :]),
            torch.from_numpy(genfromtxt(
                args.token_file + '-re-calculated-20-probs.csv', delimiter=','
            )[Tindex, :]),
            torch.from_numpy(genfromtxt(
                args.token_file + '-re-calculated-40-probs.csv', delimiter=','
            )[Tindex, :]),
            torch.from_numpy(genfromtxt(
                args.token_file + '-re-calculated-60-probs.csv', delimiter=','
            )[Tindex, :]),
            torch.from_numpy(genfromtxt(
                args.token_file + '-re-calculated-80-probs.csv', delimiter=','
            )[Tindex, :]),
            torch.from_numpy(genfromtxt(
                args.token_file + '-re-calculated-90-probs.csv', delimiter=','
            )[Tindex, :]),
            torch.from_numpy(genfromtxt(
                args.token_file + '-re-calculated-96-probs.csv', delimiter=','
            )[Tindex, :]),
            torch.from_numpy(genfromtxt(
                args.token_file + '-re-calculated-98-probs.csv', delimiter=','
            )[Tindex, :])
        ]
    )


t1 = time.time()

# csv_saves = []
# csvWriters = []
# if args.method == "transform":
#     csv_saves.append(open(args.token_file + '-' + str(args.k) + '-detect/watermarked-' +
#                      str(args.Tindex) + '.csv', 'w'))
#     csvWriters.append(csv.writer(csv_saves[-1], delimiter=','))
# elif args.method == "gumbel":
#     csv_saves.append(open(args.token_file + '-' + str(args.k) + '-detect/watermarked-' +
#                      str(args.Tindex) + '.csv', 'w'))
#     csvWriters.append(csv.writer(csv_saves[-1], delimiter=','))
# else:
#     raise

watermarked_sample = watermarked_samples[Tindex, :]

t0 = time.time()
watermarked_pval = test(watermarked_sample, seeds[Tindex], test_stats)
log_file.write(f'Ran watermarked test in (t = {time.time()-t0} seconds)\n')
log_file.flush()
np.savetxt(
    args.token_file + '-' + str(args.k) + '-detect/watermarked-' +
                     str(args.Tindex) + '.csv', watermarked_pval, delimiter=','
)
# csvWriters[0].writerow(np.asarray(watermarked_pval))
# csv_saves[0].flush()
log_file.write(args.token_file + '/' + str(args.Tindex) + ' done')
log_file.write(f'Ran the experiment (t = {time.time()-t1} seconds)\n')
log_file.close()

# for csv_save in csv_saves:
#     csv_save.close()
