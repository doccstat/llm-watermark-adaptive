import time
import torch

from collections import defaultdict
import copy

import numpy as np
from numpy import genfromtxt

from watermarking.detection import permutation_test, phi

from watermarking.transform.score import transform_score, transform_edit_score
from watermarking.transform.score import its_score, itsl_score
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
    def dist1(x, y): return transform_edit_score(x, y, gamma=args.gamma)

    def test_stat1(
        tokens, n, k, generator, vocab_size, null=False
    ):
        return phi(
            tokens=tokens,
            n=n,
            k=k,
            generator=generator,
            key_func=transform_key_func,
            vocab_size=vocab_size,
            dist=dist1,
            null=False,
            normalize=True
        )
    test_stats.append(test_stat1)
    def dist2(x, y): return transform_score(x, y)

    def test_stat2(
        tokens, n, k, generator, vocab_size, null=False
    ):
        return phi(
            tokens=tokens,
            n=n,
            k=k,
            generator=generator,
            key_func=transform_key_func,
            vocab_size=vocab_size,
            dist=dist2,
            null=False,
            normalize=True
        )
    test_stats.append(test_stat2)
    def dist3(x, y): return its_score(x, y, vocab_size=vocab_size)

    def test_stat3(
        tokens, n, k, generator, vocab_size, null=False
    ):
        return phi(
            tokens=tokens,
            n=n,
            k=k,
            generator=generator,
            key_func=transform_key_func,
            vocab_size=vocab_size,
            dist=dist3,
            null=False,
            normalize=True
        )
    test_stats.append(test_stat3)

    def dist4(x, y): return itsl_score(
        x, y, vocab_size=vocab_size, gamma=args.gamma)

    def test_stat4(
        tokens, n, k, generator, vocab_size, null=False
    ):
        return phi(
            tokens=tokens,
            n=n,
            k=k,
            generator=generator,
            key_func=transform_key_func,
            vocab_size=vocab_size,
            dist=dist4,
            null=False,
            normalize=True
        )
    test_stats.append(test_stat4)


elif args.method == "gumbel":
    test_stats = []

    true_probs = torch.from_numpy(genfromtxt(
        args.token_file + '-probs.csv', delimiter=','
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
            normalize=False
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
            normalize=False
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
            normalize=False
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
            normalize=False
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
            normalize=False
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
            normalize=False
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
            normalize=False
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
            normalize=False
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
            normalize=False
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
            normalize=False
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
            normalize=False
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
            normalize=False
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
            normalize=False
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
            normalize=False
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
            normalize=False
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
            normalize=False
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
            normalize=False
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
            normalize=False
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
            normalize=False
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
            normalize=False
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
            normalize=False
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
            normalize=False
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
            normalize=False
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
            normalize=False
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
            normalize=False
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
            normalize=False
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
            normalize=False
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
            normalize=False
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
            normalize=False
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
            normalize=False
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
            normalize=False
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
            normalize=False
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
            normalize=False
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
            normalize=False
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
            normalize=False
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
            normalize=False
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
            normalize=False
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
            normalize=False
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
            normalize=False
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
            normalize=False
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
            normalize=False
        )
    test_stats.append(test_stat_ems_adaptive_icl_13)

    # empty_probs_exp_1 = genfromtxt(
    #     args.token_file + '-re-calculated-empty-probs.csv', delimiter=','
    # )[Tindex, :]
    # empty_probs_exp_1[0] = 0.5
    # empty_probs_exp_1 = torch.from_numpy(empty_probs_exp_1)

    # def metric_ems_adaptive_empty_exp_1(x, y, probs):
    #     return ems_adaptive(x, y, probs, 1.0, 0.0, 0.0, 1.0)

    # def test_stat_ems_adaptive_empty_exp_1(
    #     tokens, n, k, generator, vocab_size, null=False
    # ):
    #     return phi(
    #         tokens=tokens,
    #         n=n,
    #         k=k,
    #         generator=generator,
    #         key_func=gumbel_key_func,
    #         vocab_size=vocab_size,
    #         dist=metric_ems_adaptive_empty_exp_1,
    #         empty_probs=empty_probs_exp_1,
    #         null=null,
    #         normalize=False
    #     )
    # test_stats.append(test_stat_ems_adaptive_empty_exp_1)

    # empty_probs_exp_2 = genfromtxt(
    #     args.token_file + '-re-calculated-empty-probs.csv', delimiter=','
    # )[Tindex, :]
    # empty_probs_exp_2[0] = 1.0
    # empty_probs_exp_2 = torch.from_numpy(empty_probs_exp_2)

    # def metric_ems_adaptive_empty_exp_2(x, y, probs):
    #     return ems_adaptive(x, y, probs, 1.0, 0.0, 0.0, 1.0)

    # def test_stat_ems_adaptive_empty_exp_2(
    #     tokens, n, k, generator, vocab_size, null=False
    # ):
    #     return phi(
    #         tokens=tokens,
    #         n=n,
    #         k=k,
    #         generator=generator,
    #         key_func=gumbel_key_func,
    #         vocab_size=vocab_size,
    #         dist=metric_ems_adaptive_empty_exp_2,
    #         empty_probs=empty_probs_exp_2,
    #         null=null,
    #         normalize=False
    #     )
    # test_stats.append(test_stat_ems_adaptive_empty_exp_2)

    # empty_probs_exp_3 = genfromtxt(
    #     args.token_file + '-re-calculated-empty-probs.csv', delimiter=','
    # )[Tindex, :]
    # empty_probs_exp_3 = np.minimum(
    #     np.where(
    #         empty_probs_exp_3 >= 0.1, 3 * empty_probs_exp_3, empty_probs_exp_3),
    #     1.0
    # )
    # empty_probs_exp_3 = torch.from_numpy(empty_probs_exp_3)

    # def metric_ems_adaptive_empty_exp_3(x, y, probs):
    #     return ems_adaptive(x, y, probs, 1.0, 0.0, 0.0, 1.0)

    # def test_stat_ems_adaptive_empty_exp_3(
    #     tokens, n, k, generator, vocab_size, null=False
    # ):
    #     return phi(
    #         tokens=tokens,
    #         n=n,
    #         k=k,
    #         generator=generator,
    #         key_func=gumbel_key_func,
    #         vocab_size=vocab_size,
    #         dist=metric_ems_adaptive_empty_exp_3,
    #         empty_probs=empty_probs_exp_3,
    #         null=null,
    #         normalize=False
    #     )
    # test_stats.append(test_stat_ems_adaptive_empty_exp_3)

    # empty_probs_exp_4 = genfromtxt(
    #     args.token_file + '-re-calculated-empty-probs.csv', delimiter=','
    # )[Tindex, :]
    # empty_probs_exp_4 = np.minimum(
    #     np.where(
    #         empty_probs_exp_4 >= 0.1, 2 * empty_probs_exp_4, empty_probs_exp_4
    #     ),
    #     1.0
    # )
    # empty_probs_exp_4 = torch.from_numpy(empty_probs_exp_4)

    # def metric_ems_adaptive_empty_exp_4(x, y, probs):
    #     return ems_adaptive(x, y, probs, 1.0, 0.0, 0.0, 1.0)

    # def test_stat_ems_adaptive_empty_exp_4(
    #     tokens, n, k, generator, vocab_size, null=False
    # ):
    #     return phi(
    #         tokens=tokens,
    #         n=n,
    #         k=k,
    #         generator=generator,
    #         key_func=gumbel_key_func,
    #         vocab_size=vocab_size,
    #         dist=metric_ems_adaptive_empty_exp_4,
    #         empty_probs=empty_probs_exp_4,
    #         null=null,
    #         normalize=False
    #     )
    # test_stats.append(test_stat_ems_adaptive_empty_exp_4)

    # empty_probs_exp_5 = genfromtxt(
    #     args.token_file + '-re-calculated-empty-probs.csv', delimiter=','
    # )[Tindex, :]
    # empty_probs_exp_5 = 0.8 * empty_probs_exp_5 + 0.2
    # empty_probs_exp_5 = torch.from_numpy(empty_probs_exp_5)

    # def metric_ems_adaptive_empty_exp_5(x, y, probs):
    #     return ems_adaptive(x, y, probs, 1.0, 0.0, 0.0, 1.0)

    # def test_stat_ems_adaptive_empty_exp_5(
    #     tokens, n, k, generator, vocab_size, null=False
    # ):
    #     return phi(
    #         tokens=tokens,
    #         n=n,
    #         k=k,
    #         generator=generator,
    #         key_func=gumbel_key_func,
    #         vocab_size=vocab_size,
    #         dist=metric_ems_adaptive_empty_exp_5,
    #         empty_probs=empty_probs_exp_5,
    #         null=null,
    #         normalize=False
    #     )
    # test_stats.append(test_stat_ems_adaptive_empty_exp_5)

    # empty_probs_exp_6 = genfromtxt(
    #     args.token_file + '-re-calculated-empty-probs.csv', delimiter=','
    # )[Tindex, :]
    # empty_probs_exp_6 = np.where(
    #     empty_probs_exp_6 >= 0.1,
    #     0.8 * empty_probs_exp_6 + 0.2,
    #     empty_probs_exp_6
    # )
    # empty_probs_exp_6 = torch.from_numpy(empty_probs_exp_6)

    # def metric_ems_adaptive_empty_exp_6(x, y, probs):
    #     return ems_adaptive(x, y, probs, 1.0, 0.0, 0.0, 1.0)

    # def test_stat_ems_adaptive_empty_exp_6(
    #     tokens, n, k, generator, vocab_size, null=False
    # ):
    #     return phi(
    #         tokens=tokens,
    #         n=n,
    #         k=k,
    #         generator=generator,
    #         key_func=gumbel_key_func,
    #         vocab_size=vocab_size,
    #         dist=metric_ems_adaptive_empty_exp_6,
    #         empty_probs=empty_probs_exp_6,
    #         null=null,
    #         normalize=False
    #     )
    # test_stats.append(test_stat_ems_adaptive_empty_exp_6)

    # empty_probs_exp_7 = genfromtxt(
    #     args.token_file + '-re-calculated-empty-probs.csv', delimiter=','
    # )[Tindex, :]
    # empty_probs_exp_7 = np.where(
    #     empty_probs_exp_7 <= 1e-6, 0.9, empty_probs_exp_7)
    # empty_probs_exp_7 = torch.from_numpy(empty_probs_exp_7)

    # def metric_ems_adaptive_empty_exp_7(x, y, probs):
    #     return ems_adaptive(x, y, probs, 1.0, 0.0, 0.0, 1.0)

    # def test_stat_ems_adaptive_empty_exp_7(
    #     tokens, n, k, generator, vocab_size, null=False
    # ):
    #     return phi(
    #         tokens=tokens,
    #         n=n,
    #         k=k,
    #         generator=generator,
    #         key_func=gumbel_key_func,
    #         vocab_size=vocab_size,
    #         dist=metric_ems_adaptive_empty_exp_7,
    #         empty_probs=empty_probs_exp_7,
    #         null=null,
    #         normalize=False
    #     )
    # test_stats.append(test_stat_ems_adaptive_empty_exp_7)

    # empty_probs_exp_8 = genfromtxt(
    #     args.token_file + '-re-calculated-empty-probs.csv', delimiter=','
    # )[Tindex, :]
    # empty_probs_exp_8 = np.sqrt(empty_probs_exp_8)
    # empty_probs_exp_8 = torch.from_numpy(empty_probs_exp_8)

    # def metric_ems_adaptive_empty_exp_8(x, y, probs):
    #     return ems_adaptive(x, y, probs, 1.0, 0.0, 0.0, 1.0)

    # def test_stat_ems_adaptive_empty_exp_8(
    #     tokens, n, k, generator, vocab_size, null=False
    # ):
    #     return phi(
    #         tokens=tokens,
    #         n=n,
    #         k=k,
    #         generator=generator,
    #         key_func=gumbel_key_func,
    #         vocab_size=vocab_size,
    #         dist=metric_ems_adaptive_empty_exp_8,
    #         empty_probs=empty_probs_exp_8,
    #         null=null,
    #         normalize=False
    #     )
    # test_stats.append(test_stat_ems_adaptive_empty_exp_8)

    # empty_probs_exp_9 = genfromtxt(
    #     args.token_file + '-re-calculated-empty-probs.csv', delimiter=','
    # )[Tindex, :]
    # empty_probs_exp_9 = np.power(empty_probs_exp_9, 1/3)
    # empty_probs_exp_9 = torch.from_numpy(empty_probs_exp_9)

    # def metric_ems_adaptive_empty_exp_9(x, y, probs):
    #     return ems_adaptive(x, y, probs, 1.0, 0.0, 0.0, 1.0)

    # def test_stat_ems_adaptive_empty_exp_9(
    #     tokens, n, k, generator, vocab_size, null=False
    # ):
    #     return phi(
    #         tokens=tokens,
    #         n=n,
    #         k=k,
    #         generator=generator,
    #         key_func=gumbel_key_func,
    #         vocab_size=vocab_size,
    #         dist=metric_ems_adaptive_empty_exp_9,
    #         empty_probs=empty_probs_exp_9,
    #         null=null,
    #         normalize=False
    #     )
    # test_stats.append(test_stat_ems_adaptive_empty_exp_9)

    # # True probabilities with values less than 0.5 set to 0.5
    # big_true_probs = genfromtxt(
    #     args.token_file + '-probs.csv', delimiter=','
    # )[Tindex, :]
    # big_true_probs = np.maximum(big_true_probs, 0.5)
    # big_true_probs = torch.from_numpy(big_true_probs)

    # def metric_ems_adaptive_big_true(x, y, probs):
    #     return ems_adaptive(x, y, probs, 1.0, 0.0, 0.0, 1.0)

    # def test_stat_ems_adaptive_big_true(
    #     tokens, n, k, generator, vocab_size, null=False
    # ):
    #     return phi(
    #         tokens=tokens,
    #         n=n,
    #         k=k,
    #         generator=generator,
    #         key_func=gumbel_key_func,
    #         vocab_size=vocab_size,
    #         dist=metric_ems_adaptive_big_true,
    #         empty_probs=big_true_probs,
    #         null=null,
    #         normalize=False
    #     )
    # test_stats.append(test_stat_ems_adaptive_big_true)

    # # True probabilities with values greater than 0.5 set to 0.5
    # small_true_probs = genfromtxt(
    #     args.token_file + '-probs.csv', delimiter=','
    # )[Tindex, :]
    # small_true_probs = np.minimum(small_true_probs, 0.5)
    # small_true_probs = torch.from_numpy(small_true_probs)

    # def metric_ems_adaptive_small_true(x, y, probs):
    #     return ems_adaptive(x, y, probs, 1.0, 0.0, 0.0, 1.0)

    # def test_stat_ems_adaptive_small_true(
    #     tokens, n, k, generator, vocab_size, null=False
    # ):
    #     return phi(
    #         tokens=tokens,
    #         n=n,
    #         k=k,
    #         generator=generator,
    #         key_func=gumbel_key_func,
    #         vocab_size=vocab_size,
    #         dist=metric_ems_adaptive_small_true,
    #         empty_probs=small_true_probs,
    #         null=null,
    #         normalize=False
    #     )
    # test_stats.append(test_stat_ems_adaptive_small_true)

    # # Empty probabilities with values less than 0.5 set to 0.5
    # big_empty_probs = genfromtxt(
    #     args.token_file + '-re-calculated-empty-probs.csv', delimiter=','
    # )[Tindex, :]
    # big_empty_probs = np.maximum(big_empty_probs, 0.5)
    # big_empty_probs = torch.from_numpy(big_empty_probs)

    # def metric_ems_adaptive_big_empty(x, y, probs):
    #     return ems_adaptive(x, y, probs, 1.0, 0.0, 0.0, 1.0)

    # def test_stat_ems_adaptive_big_empty(
    #     tokens, n, k, generator, vocab_size, null=False
    # ):
    #     return phi(
    #         tokens=tokens,
    #         n=n,
    #         k=k,
    #         generator=generator,
    #         key_func=gumbel_key_func,
    #         vocab_size=vocab_size,
    #         dist=metric_ems_adaptive_big_empty,
    #         empty_probs=big_empty_probs,
    #         null=null,
    #         normalize=False
    #     )
    # test_stats.append(test_stat_ems_adaptive_big_empty)

    # # Empty probabilities with values greater than 0.5 set to 0.5
    # small_empty_probs = genfromtxt(
    #     args.token_file + '-re-calculated-empty-probs.csv', delimiter=','
    # )[Tindex, :]
    # small_empty_probs = np.minimum(small_empty_probs, 0.5)
    # small_empty_probs = torch.from_numpy(small_empty_probs)

    # def metric_ems_adaptive_small_empty(x, y, probs):
    #     return ems_adaptive(x, y, probs, 1.0, 0.0, 0.0, 1.0)

    # def test_stat_ems_adaptive_small_empty(
    #     tokens, n, k, generator, vocab_size, null=False
    # ):
    #     return phi(
    #         tokens=tokens,
    #         n=n,
    #         k=k,
    #         generator=generator,
    #         key_func=gumbel_key_func,
    #         vocab_size=vocab_size,
    #         dist=metric_ems_adaptive_small_empty,
    #         empty_probs=small_empty_probs,
    #         null=null,
    #         normalize=False
    #     )
    # test_stats.append(test_stat_ems_adaptive_small_empty)

    # # True probabilities with smallest value set to 0.5
    # big_true_probs_1 = genfromtxt(
    #     args.token_file + '-probs.csv', delimiter=','
    # )[Tindex, :]
    # big_true_probs_1[big_true_probs_1.argmin()] = 0.5
    # big_true_probs_1 = torch.from_numpy(big_true_probs_1)

    # def metric_ems_adaptive_big_true_1(x, y, probs):
    #     return ems_adaptive(x, y, probs, 1.0, 0.0, 0.0, 1.0)

    # def test_stat_ems_adaptive_big_true_1(
    #     tokens, n, k, generator, vocab_size, null=False
    # ):
    #     return phi(
    #         tokens=tokens,
    #         n=n,
    #         k=k,
    #         generator=generator,
    #         key_func=gumbel_key_func,
    #         vocab_size=vocab_size,
    #         dist=metric_ems_adaptive_big_true_1,
    #         empty_probs=big_true_probs_1,
    #         null=null,
    #         normalize=False
    #     )
    # test_stats.append(test_stat_ems_adaptive_big_true_1)

    # # True probabilities with smallest two values set to 0.5
    # big_true_probs_2 = genfromtxt(
    #     args.token_file + '-probs.csv', delimiter=','
    # )[Tindex, :]
    # big_true_probs_2[big_true_probs_2.argsort()[:2]] = 0.5
    # big_true_probs_2 = torch.from_numpy(big_true_probs_2)

    # def metric_ems_adaptive_big_true_2(x, y, probs):
    #     return ems_adaptive(x, y, probs, 1.0, 0.0, 0.0, 1.0)

    # def test_stat_ems_adaptive_big_true_2(
    #     tokens, n, k, generator, vocab_size, null=False
    # ):
    #     return phi(
    #         tokens=tokens,
    #         n=n,
    #         k=k,
    #         generator=generator,
    #         key_func=gumbel_key_func,
    #         vocab_size=vocab_size,
    #         dist=metric_ems_adaptive_big_true_2,
    #         empty_probs=big_true_probs_2,
    #         null=null,
    #         normalize=False
    #     )
    # test_stats.append(test_stat_ems_adaptive_big_true_2)

    # # True probabilities with smallest three values set to 0.5
    # big_true_probs_3 = genfromtxt(
    #     args.token_file + '-probs.csv', delimiter=','
    # )[Tindex, :]
    # big_true_probs_3[big_true_probs_3.argsort()[:3]] = 0.5
    # big_true_probs_3 = torch.from_numpy(big_true_probs_3)

    # def metric_ems_adaptive_big_true_3(x, y, probs):
    #     return ems_adaptive(x, y, probs, 1.0, 0.0, 0.0, 1.0)

    # def test_stat_ems_adaptive_big_true_3(
    #     tokens, n, k, generator, vocab_size, null=False
    # ):
    #     return phi(
    #         tokens=tokens,
    #         n=n,
    #         k=k,
    #         generator=generator,
    #         key_func=gumbel_key_func,
    #         vocab_size=vocab_size,
    #         dist=metric_ems_adaptive_big_true_3,
    #         empty_probs=big_true_probs_3,
    #         null=null,
    #         normalize=False
    #     )
    # test_stats.append(test_stat_ems_adaptive_big_true_3)

    # # True probabilities with smallest four values set to 0.5
    # big_true_probs_4 = genfromtxt(
    #     args.token_file + '-probs.csv', delimiter=','
    # )[Tindex, :]
    # big_true_probs_4[big_true_probs_4.argsort()[:4]] = 0.5
    # big_true_probs_4 = torch.from_numpy(big_true_probs_4)

    # def metric_ems_adaptive_big_true_4(x, y, probs):
    #     return ems_adaptive(x, y, probs, 1.0, 0.0, 0.0, 1.0)

    # def test_stat_ems_adaptive_big_true_4(
    #     tokens, n, k, generator, vocab_size, null=False
    # ):
    #     return phi(
    #         tokens=tokens,
    #         n=n,
    #         k=k,
    #         generator=generator,
    #         key_func=gumbel_key_func,
    #         vocab_size=vocab_size,
    #         dist=metric_ems_adaptive_big_true_4,
    #         empty_probs=big_true_probs_4,
    #         null=null,
    #         normalize=False
    #     )
    # test_stats.append(test_stat_ems_adaptive_big_true_4)

    # # True probabilities with smallest five values set to 0.5
    # big_true_probs_5 = genfromtxt(
    #     args.token_file + '-probs.csv', delimiter=','
    # )[Tindex, :]
    # big_true_probs_5[big_true_probs_5.argsort()[:5]] = 0.5
    # big_true_probs_5 = torch.from_numpy(big_true_probs_5)

    # def metric_ems_adaptive_big_true_5(x, y, probs):
    #     return ems_adaptive(x, y, probs, 1.0, 0.0, 0.0, 1.0)

    # def test_stat_ems_adaptive_big_true_5(
    #     tokens, n, k, generator, vocab_size, null=False
    # ):
    #     return phi(
    #         tokens=tokens,
    #         n=n,
    #         k=k,
    #         generator=generator,
    #         key_func=gumbel_key_func,
    #         vocab_size=vocab_size,
    #         dist=metric_ems_adaptive_big_true_5,
    #         empty_probs=big_true_probs_5,
    #         null=null,
    #         normalize=False
    #     )
    # test_stats.append(test_stat_ems_adaptive_big_true_5)

    # # True probabilities with all replaced by 0.5 except the smallest
    # true_probs_only_smallest = np.ones(true_probs.shape) * 0.5
    # true_probs_only_smallest[true_probs.argmin()] = true_probs.min()
    # true_probs_only_smallest = torch.from_numpy(true_probs_only_smallest)

    # def metric_ems_adaptive_true_probs_only_smallest(x, y, probs):
    #     return ems_adaptive(x, y, probs, 1.0, 0.0, 0.0, 1.0)

    # def test_stat_ems_adaptive_true_probs_only_smallest(
    #     tokens, n, k, generator, vocab_size, null=False
    # ):
    #     return phi(
    #         tokens=tokens,
    #         n=n,
    #         k=k,
    #         generator=generator,
    #         key_func=gumbel_key_func,
    #         vocab_size=vocab_size,
    #         dist=metric_ems_adaptive_true_probs_only_smallest,
    #         empty_probs=true_probs_only_smallest,
    #         null=null,
    #         normalize=False
    #     )
    # test_stats.append(test_stat_ems_adaptive_true_probs_only_smallest)

    # # True probabilities with all replaced by 1 except the smallest 10%
    # true_probs_only_smallest_1 = np.ones(true_probs.shape)
    # true_probs_only_smallest_1[true_probs.argsort()[:int(0.1 * len(true_probs))]] = true_probs.min()
    # true_probs_only_smallest_1 = torch.from_numpy(true_probs_only_smallest_1)

    # def metric_ems_adaptive_true_probs_only_smallest_1(x, y, probs):
    #     return ems_adaptive(x, y, probs, 1.0, 0.0, 0.0, 1.0)

    # def test_stat_ems_adaptive_true_probs_only_smallest_1(
    #         tokens, n, k, generator, vocab_size, null=False
    # ):
    #     return phi(
    #         tokens=tokens,
    #         n=n,
    #         k=k,
    #         generator=generator,
    #         key_func=gumbel_key_func,
    #         vocab_size=vocab_size,
    #         dist=metric_ems_adaptive_true_probs_only_smallest_1,
    #         empty_probs=true_probs_only_smallest_1,
    #         null=null,
    #         normalize=False
    #     )
    # test_stats.append(test_stat_ems_adaptive_true_probs_only_smallest_1)

    # # True probabilities with all replaced by 1 except the smallest 20%
    # true_probs_only_smallest_2 = np.ones(true_probs.shape)
    # true_probs_only_smallest_2[true_probs.argsort()[:int(0.2 * len(true_probs))]] = true_probs.min()
    # true_probs_only_smallest_2 = torch.from_numpy(true_probs_only_smallest_2)

    # def metric_ems_adaptive_true_probs_only_smallest_2(x, y, probs):
    #     return ems_adaptive(x, y, probs, 1.0, 0.0, 0.0, 1.0)

    # def test_stat_ems_adaptive_true_probs_only_smallest_2(
    #         tokens, n, k, generator, vocab_size, null=False
    # ):
    #     return phi(
    #         tokens=tokens,
    #         n=n,
    #         k=k,
    #         generator=generator,
    #         key_func=gumbel_key_func,
    #         vocab_size=vocab_size,
    #         dist=metric_ems_adaptive_true_probs_only_smallest_2,
    #         empty_probs=true_probs_only_smallest_2,
    #         null=null,
    #         normalize=False
    #     )
    # test_stats.append(test_stat_ems_adaptive_true_probs_only_smallest_2)

    # Empty probabilities with all replaced by 1 except the smallest 10%
    empty_probs_only_smallest = np.ones(empty_probs.shape)
    empty_probs_only_smallest[empty_probs.argsort()[:int(0.1 * len(empty_probs))]] = empty_probs.min()
    empty_probs_only_smallest = torch.from_numpy(empty_probs_only_smallest)

    def metric_ems_adaptive_empty_probs_only_smallest(x, y, probs):
        return ems_adaptive(x, y, probs, 1.0, 0.0, 0.0, 1.0)

    def test_stat_ems_adaptive_empty_probs_only_smallest(
            tokens, n, k, generator, vocab_size, null=False
    ):
        return phi(
            tokens=tokens,
            n=n,
            k=k,
            generator=generator,
            key_func=gumbel_key_func,
            vocab_size=vocab_size,
            dist=metric_ems_adaptive_empty_probs_only_smallest,
            empty_probs=empty_probs_only_smallest,
            null=null,
            normalize=False
        )
    test_stats.append(test_stat_ems_adaptive_empty_probs_only_smallest)

    # Empty probabilities with all replaced by 1 except the smallest 20%
    empty_probs_only_smallest_2 = np.ones(empty_probs.shape)
    empty_probs_only_smallest_2[empty_probs.argsort()[:int(0.2 * len(empty_probs))]] = empty_probs.min()
    empty_probs_only_smallest_2 = torch.from_numpy(empty_probs_only_smallest_2)

    def metric_ems_adaptive_empty_probs_only_smallest_2(x, y, probs):
        return ems_adaptive(x, y, probs, 1.0, 0.0, 0.0, 1.0)

    def test_stat_ems_adaptive_empty_probs_only_smallest_2(
            tokens, n, k, generator, vocab_size, null=False
    ):
        return phi(
            tokens=tokens,
            n=n,
            k=k,
            generator=generator,
            key_func=gumbel_key_func,
            vocab_size=vocab_size,
            dist=metric_ems_adaptive_empty_probs_only_smallest_2,
            empty_probs=empty_probs_only_smallest_2,
            null=null,
            normalize=False
        )
    test_stats.append(test_stat_ems_adaptive_empty_probs_only_smallest_2)

    # Empty probabilities with all replaced by 1 except the smallest 30%
    empty_probs_only_smallest_3 = np.ones(empty_probs.shape)
    empty_probs_only_smallest_3[empty_probs.argsort()[:int(0.3 * len(empty_probs))]] = empty_probs.min()
    empty_probs_only_smallest_3 = torch.from_numpy(empty_probs_only_smallest_3)

    def metric_ems_adaptive_empty_probs_only_smallest_3(x, y, probs):
        return ems_adaptive(x, y, probs, 1.0, 0.0, 0.0, 1.0)

    def test_stat_ems_adaptive_empty_probs_only_smallest_3(
            tokens, n, k, generator, vocab_size, null=False
    ):
        return phi(
            tokens=tokens,
            n=n,
            k=k,
            generator=generator,
            key_func=gumbel_key_func,
            vocab_size=vocab_size,
            dist=metric_ems_adaptive_empty_probs_only_smallest_3,
            empty_probs=empty_probs_only_smallest_3,
            null=null,
            normalize=False
        )
    test_stats.append(test_stat_ems_adaptive_empty_probs_only_smallest_3)

    # Empty probabilities with all replaced by 1 except the smallest 40%
    empty_probs_only_smallest_4 = np.ones(empty_probs.shape)
    empty_probs_only_smallest_4[empty_probs.argsort()[:int(0.4 * len(empty_probs))]] = empty_probs.min()
    empty_probs_only_smallest_4 = torch.from_numpy(empty_probs_only_smallest_4)

    def metric_ems_adaptive_empty_probs_only_smallest_4(x, y, probs):
        return ems_adaptive(x, y, probs, 1.0, 0.0, 0.0, 1.0)

    def test_stat_ems_adaptive_empty_probs_only_smallest_4(
            tokens, n, k, generator, vocab_size, null=False
    ):
        return phi(
            tokens=tokens,
            n=n,
            k=k,
            generator=generator,
            key_func=gumbel_key_func,
            vocab_size=vocab_size,
            dist=metric_ems_adaptive_empty_probs_only_smallest_4,
            empty_probs=empty_probs_only_smallest_4,
            null=null,
            normalize=False
        )
    test_stats.append(test_stat_ems_adaptive_empty_probs_only_smallest_4)

    # Probabilities calculated with 20% prompt (80% modified)
    probs_20 = genfromtxt(
        args.token_file + '-re-calculated-20-probs.csv', delimiter=','
    )[Tindex, :]
    probs_20 = torch.from_numpy(probs_20)

    def metric_ems_adaptive_probs_20(x, y, probs):
        return ems_adaptive(x, y, probs, 1.0, 0.0, 0.0, 1.0)

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
            normalize=False
        )
    test_stats.append(test_stat_ems_adaptive_probs_20)

    # Probabilities calculated with 40% prompt (60% modified)
    probs_40 = genfromtxt(
        args.token_file + '-re-calculated-40-probs.csv', delimiter=','
    )[Tindex, :]
    probs_40 = torch.from_numpy(probs_40)

    def metric_ems_adaptive_probs_40(x, y, probs):
        return ems_adaptive(x, y, probs, 1.0, 0.0, 0.0, 1.0)

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
            normalize=False
        )
    test_stats.append(test_stat_ems_adaptive_probs_40)

    # Probabilities calculated with 60% prompt (40% modified)
    probs_60 = genfromtxt(
        args.token_file + '-re-calculated-60-probs.csv', delimiter=','
    )[Tindex, :]
    probs_60 = torch.from_numpy(probs_60)

    def metric_ems_adaptive_probs_60(x, y, probs):
        return ems_adaptive(x, y, probs, 1.0, 0.0, 0.0, 1.0)

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
            normalize=False
        )
    test_stats.append(test_stat_ems_adaptive_probs_60)

    # Probabilities calculated with 80% prompt (20% modified)
    probs_80 = genfromtxt(
        args.token_file + '-re-calculated-80-probs.csv', delimiter=','
    )[Tindex, :]
    probs_80 = torch.from_numpy(probs_80)

    def metric_ems_adaptive_probs_80(x, y, probs):
        return ems_adaptive(x, y, probs, 1.0, 0.0, 0.0, 1.0)

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
            normalize=False
        )
    test_stats.append(test_stat_ems_adaptive_probs_80)

    # Probabilities calculated with 90% prompt (10% modified)
    probs_90 = genfromtxt(
        args.token_file + '-re-calculated-90-probs.csv', delimiter=','
    )[Tindex, :]
    probs_90 = torch.from_numpy(probs_90)

    def metric_ems_adaptive_probs_90(x, y, probs):
        return ems_adaptive(x, y, probs, 1.0, 0.0, 0.0, 1.0)

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
            normalize=False
        )
    test_stats.append(test_stat_ems_adaptive_probs_90)

    # Probabilities calculated with 96% prompt (4% modified)
    probs_96 = genfromtxt(
        args.token_file + '-re-calculated-96-probs.csv', delimiter=','
    )[Tindex, :]
    probs_96 = torch.from_numpy(probs_96)

    def metric_ems_adaptive_probs_96(x, y, probs):
        return ems_adaptive(x, y, probs, 1.0, 0.0, 0.0, 1.0)

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
            normalize=False
        )
    test_stats.append(test_stat_ems_adaptive_probs_96)

    # Probabilities calculated with 98% prompt (2% modified)
    probs_98 = genfromtxt(
        args.token_file + '-re-calculated-98-probs.csv', delimiter=','
    )[Tindex, :]
    probs_98 = torch.from_numpy(probs_98)

    def metric_ems_adaptive_probs_98(x, y, probs):
        return ems_adaptive(x, y, probs, 1.0, 0.0, 0.0, 1.0)

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
            normalize=False
        )
    test_stats.append(test_stat_ems_adaptive_probs_98)

    # Probabilities calculated with 20% prompt (80% modified), only calculated for 80% of the tokens
    probs_20_08 = genfromtxt(
        args.token_file + '-re-calculated-20-probs.csv', delimiter=','
    )[Tindex, :]
    probs_20_08 = torch.from_numpy(probs_20_08)

    def metric_ems_adaptive_probs_20_08(x, y, probs):
        return ems_adaptive(x, y, probs, 0.8, 0.0, 0.0, 1.0)

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
            normalize=False
        )
    test_stats.append(test_stat_ems_adaptive_probs_20_08)

    # Probabilities calculated with 40% prompt (60% modified), only calculated for 80% of the tokens
    probs_40_08 = genfromtxt(
        args.token_file + '-re-calculated-40-probs.csv', delimiter=','
    )[Tindex, :]
    probs_40_08 = torch.from_numpy(probs_40_08)

    def metric_ems_adaptive_probs_40_08(x, y, probs):
        return ems_adaptive(x, y, probs, 0.8, 0.0, 0.0, 1.0)

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
            normalize=False
        )
    test_stats.append(test_stat_ems_adaptive_probs_40_08)

    # Probabilities calculated with 60% prompt (40% modified), only calculated for 80% of the tokens
    probs_60_08 = genfromtxt(
        args.token_file + '-re-calculated-60-probs.csv', delimiter=','
    )[Tindex, :]
    probs_60_08 = torch.from_numpy(probs_60_08)

    def metric_ems_adaptive_probs_60_08(x, y, probs):
        return ems_adaptive(x, y, probs, 0.8, 0.0, 0.0, 1.0)

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
            normalize=False
        )
    test_stats.append(test_stat_ems_adaptive_probs_60_08)

    # Probabilities calculated with 80% prompt (20% modified), only calculated for 80% of the tokens
    probs_80_08 = genfromtxt(
        args.token_file + '-re-calculated-80-probs.csv', delimiter=','
    )[Tindex, :]
    probs_80_08 = torch.from_numpy(probs_80_08)

    def metric_ems_adaptive_probs_80_08(x, y, probs):
        return ems_adaptive(x, y, probs, 0.8, 0.0, 0.0, 1.0)

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
            normalize=False
        )
    test_stats.append(test_stat_ems_adaptive_probs_80_08)

    # Probabilities calculated with 90% prompt (10% modified), only calculated for 80% of the tokens
    probs_90_08 = genfromtxt(
        args.token_file + '-re-calculated-90-probs.csv', delimiter=','
    )[Tindex, :]
    probs_90_08 = torch.from_numpy(probs_90_08)

    def metric_ems_adaptive_probs_90_08(x, y, probs):
        return ems_adaptive(x, y, probs, 0.8, 0.0, 0.0, 1.0)

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
            normalize=False
        )
    test_stats.append(test_stat_ems_adaptive_probs_90_08)

    # Probabilities calculated with 96% prompt (4% modified), only calculated for 80% of the tokens
    probs_96_08 = genfromtxt(
        args.token_file + '-re-calculated-96-probs.csv', delimiter=','
    )[Tindex, :]
    probs_96_08 = torch.from_numpy(probs_96_08)

    def metric_ems_adaptive_probs_96_08(x, y, probs):
        return ems_adaptive(x, y, probs, 0.8, 0.0, 0.0, 1.0)

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
            normalize=False
        )
    test_stats.append(test_stat_ems_adaptive_probs_96_08)

    # Probabilities calculated with 98% prompt (2% modified), only calculated for 80% of the tokens
    probs_98_08 = genfromtxt(
        args.token_file + '-re-calculated-98-probs.csv', delimiter=','
    )[Tindex, :]
    probs_98_08 = torch.from_numpy(probs_98_08)

    def metric_ems_adaptive_probs_98_08(x, y, probs):
        return ems_adaptive(x, y, probs, 0.8, 0.0, 0.0, 1.0)

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
            normalize=False
        )
    test_stats.append(test_stat_ems_adaptive_probs_98_08)

else:
    raise

# Don't forget to remove the folder following the readme file,
# if the experiment needs re-running.
if os.path.exists(f"{args.token_file}-detect/watermarked-{args.Tindex}.csv"):
    with open(
        f"{args.token_file}-detect/watermarked-{args.Tindex}.csv", 'r'
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


t1 = time.time()

csv_saves = []
csvWriters = []
if args.method == "transform":
    csv_saves.append(open(args.token_file + '-detect/' +
                     str(args.Tindex) + '-transform-edit.csv',
                     'w'))
    csvWriters.append(csv.writer(csv_saves[-1], delimiter=','))
    csv_saves.append(open(args.token_file + '-detect/' +
                     str(args.Tindex) + '-transform.csv',
                     'w'))
    csvWriters.append(csv.writer(csv_saves[-1], delimiter=','))
    csv_saves.append(open(args.token_file + '-detect/' +
                     str(args.Tindex) + '-its.csv',
        'w'))
    csvWriters.append(csv.writer(csv_saves[-1], delimiter=','))
    csv_saves.append(open(args.token_file + '-detect/' +
                          str(args.Tindex) + '-itsl.csv',
                          'w'))
    csvWriters.append(csv.writer(csv_saves[-1], delimiter=','))
elif args.method == "gumbel":
    csv_saves.append(open(args.token_file + '-detect/watermarked-' +
                     str(args.Tindex) + '.csv', 'w'))
    csvWriters.append(csv.writer(csv_saves[-1], delimiter=','))
    # csv_saves.append(open(args.token_file + '-detect/null-' +
    #                  str(args.Tindex) + '.csv', 'w'))
    # csvWriters.append(csv.writer(csv_saves[-1], delimiter=','))
else:
    raise

watermarked_sample = watermarked_samples[Tindex, :]
# null_sample = null_samples[Tindex, :]

t0 = time.time()
watermarked_pval = test(watermarked_sample, seeds[Tindex], test_stats)
# [test_stats[i] for i in [0, 1, 2, 3, 4, 5]]
log_file.write(f'Ran watermarked test in (t = {time.time()-t0} seconds)\n')
log_file.flush()
# t0 = time.time()
# null_pval = test(null_sample, seeds[Tindex], [
#                  test_stats[i] for i in [0, 5]])
# log_file.write(f'Ran null test in (t = {time.time()-t0} seconds)\n')
# log_file.flush()
csvWriters[0].writerow(np.asarray(watermarked_pval))
csv_saves[0].flush()
# for distance_index in range(len(null_pval)):
#     csvWriters[distance_index + len(watermarked_pval)
#                ].writerow(np.asarray(null_pval[distance_index, ]))
#     csv_saves[distance_index + len(watermarked_pval)].flush()
log_file.write(args.token_file + '/' + str(args.Tindex) + ' done')
log_file.write(f'Ran the experiment (t = {time.time()-t1} seconds)\n')
log_file.close()

for csv_save in csv_saves:
    csv_save.close()
