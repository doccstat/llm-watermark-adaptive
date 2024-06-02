import time
import torch
from datasets import load_dataset

from collections import defaultdict
import copy

import numpy as np
from numpy import genfromtxt

from watermarking.detection import adjacency

from watermarking.transform.score import transform_score, transform_edit_score
from watermarking.transform.score import its_score, itsl_score
from watermarking.transform.key import transform_key_func

from watermarking.gumbel.score import gumbel_score, gumbel_edit_score
from watermarking.gumbel.score import ems_score, emsl_score
from watermarking.gumbel.score import ems_not_adaptive_score, ems_yes_adaptive_score
from watermarking.gumbel.key import gumbel_key_func

import argparse

import csv

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
parser.add_argument('--max_seed', default=100000, type=int)
parser.add_argument('--offset', action='store_true')

parser.add_argument('--norm', default=1, type=int)
parser.add_argument('--gamma', default=0.4, type=float)
# parser.add_argument('--edit', action='store_true')
parser.add_argument('--nowatermark', action='store_true')

parser.add_argument('--deletion', default=0.0, type=float)
parser.add_argument('--insertion', default=0.0, type=float)
parser.add_argument('--substitution', default=0.0, type=float)

parser.add_argument('--kirch_gamma', default=0.25, type=float)
parser.add_argument('--kirch_delta', default=1.0, type=float)

parser.add_argument('--truncate_vocab', default=8, type=int)

args = parser.parse_args()
results['args'] = copy.deepcopy(args)

log_file = open('log/' + str(args.Tindex) + '.log', 'w')
log_file.write(str(args) + '\n')
log_file.flush()

t0 = time.time()

if args.model == "facebook/opt-1.3b":
    vocab_size = 50272
elif args.model == "openai-community/gpt2":
    vocab_size = 50257
else:
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(args.model).to(
        torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
    print(model.get_output_embeddings().weight.shape[0])
    raise
eff_vocab_size = vocab_size - args.truncate_vocab
log_file.write(f'Loaded the model (t = {time.time()-t0} seconds)\n')
log_file.flush()

while True:
    try:
        dataset = load_dataset("allenai/c4", "realnewslike",
                               split="train", streaming=True)
        break
    except:
        time.sleep(3)

prompt_tokens = args.prompt_tokens      # minimum prompt length
buffer_tokens = args.buffer_tokens
k = args.k
n = args.n     # watermark key length

seeds = np.genfromtxt(args.token_file + '-seeds.csv',
                      delimiter=',', max_rows=1)

################################################################################


def permutation_test(
    tokens, vocab_size, n, k, seed, test_stats, n_runs=100, max_seed=100000
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
    p_val = 0
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
        # assuming lower test values indicate presence of watermark
        p_val += (null_result <= test_result).float()
    null_results = np.array(null_results)

    return (np.sum(null_results <= test_results, axis=0) + 1.0) / (n_runs + 1.0)


def phi(
        tokens, n, k, generator, key_func, vocab_size, dist,
        null=False, normalize=False
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

    A = adjacency(tokens, xi, dist, k)
    closest = torch.min(A, axis=1)[0]

    return torch.min(closest)


################################################################################


watermarked_samples = genfromtxt(
    args.token_file + '-tokens-before-attack.csv', delimiter=",")
null_samples = genfromtxt(args.token_file + '-null.csv', delimiter=",")
Tindex = min(args.Tindex, watermarked_samples.shape[0])
log_file.write(f'Loaded the samples (t = {time.time()-t0} seconds)\n')
log_file.flush()


if args.method == "transform":
    test_stats = []
    def dist1(x, y): return transform_edit_score(x, y, gamma=args.gamma)

    def test_stat1(tokens, n, k, generator, vocab_size, null=False): return phi(
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

    def test_stat2(tokens, n, k, generator, vocab_size, null=False): return phi(
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

    def test_stat3(tokens, n, k, generator, vocab_size, null=False): return phi(
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

    def test_stat4(tokens, n, k, generator, vocab_size, null=False): return phi(
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
    # def dist1(x, y): return gumbel_edit_score(x, y, gamma=args.gamma)

    # def test_stat1(tokens, n, k, generator, vocab_size, null=False): return phi(
    #     tokens=tokens,
    #     n=n,
    #     k=k,
    #     generator=generator,
    #     key_func=gumbel_key_func,
    #     vocab_size=vocab_size,
    #     dist=dist1,
    #     null=null,
    #     normalize=False
    # )
    # test_stats.append(test_stat1)
    # def dist2(x, y): return gumbel_score(x, y)

    # def test_stat2(tokens, n, k, generator, vocab_size, null=False): return phi(
    #     tokens=tokens,
    #     n=n,
    #     k=k,
    #     generator=generator,
    #     key_func=gumbel_key_func,
    #     vocab_size=vocab_size,
    #     dist=dist2,
    #     null=null,
    #     normalize=False
    # )
    # test_stats.append(test_stat2)
    # def dist3(x, y): return ems_score(x, y)

    # def test_stat3(tokens, n, k, generator, vocab_size, null=False): return phi(
    #     tokens=tokens,
    #     n=n,
    #     k=k,
    #     generator=generator,
    #     key_func=gumbel_key_func,
    #     vocab_size=vocab_size,
    #     dist=dist3,
    #     null=null,
    #     normalize=False
    # )
    # test_stats.append(test_stat3)
    # def dist4(x, y): return emsl_score(x, y, gamma=args.gamma)

    # def test_stat4(tokens, n, k, generator, vocab_size, null=False): return phi(
    #     tokens=tokens,
    #     n=n,
    #     k=k,
    #     generator=generator,
    #     key_func=gumbel_key_func,
    #     vocab_size=vocab_size,
    #     dist=dist4,
    #     null=null,
    #     normalize=False
    # )
    # test_stats.append(test_stat4)
    def dist5(x, y): return ems_not_adaptive_score(x, y)

    def test_stat5(tokens, n, k, generator, vocab_size, null=False): return phi(
        tokens=tokens,
        n=n,
        k=k,
        generator=generator,
        key_func=gumbel_key_func,
        vocab_size=vocab_size,
        dist=dist5,
        null=null,
        normalize=False
    )
    test_stats.append(test_stat5)

    def dist6(x, y): return ems_yes_adaptive_score(
        x, y, torch.from_numpy(genfromtxt(args.token_file + '-probs.csv',
                                          delimiter=',')[Tindex, :])
    )

    def test_stat6(tokens, n, k, generator, vocab_size, null=False): return phi(
        tokens=tokens,
        n=n,
        k=k,
        generator=generator,
        key_func=gumbel_key_func,
        vocab_size=vocab_size,
        dist=dist6,
        null=null,
        normalize=False
    )
    test_stats.append(test_stat6)

    def dist7(x, y): return ems_yes_adaptive_score(
        x, y, torch.from_numpy(genfromtxt(args.token_file + '-empty-probs.csv',
                                          delimiter=',')[Tindex, :])
    )

    def test_stat7(tokens, n, k, generator, vocab_size, null=False): return phi(
        tokens=tokens,
        n=n,
        k=k,
        generator=generator,
        key_func=gumbel_key_func,
        vocab_size=vocab_size,
        dist=dist7,
        null=null,
        normalize=False
    )
    test_stats.append(test_stat7)

    def dist8(x, y): return ems_yes_adaptive_score(
        x, y, torch.from_numpy(genfromtxt(args.token_file + '-null-probs.csv',
                                          delimiter=',')[Tindex, :])
    )

    def test_stat8(tokens, n, k, generator, vocab_size, null=False): return phi(
        tokens=tokens,
        n=n,
        k=k,
        generator=generator,
        key_func=gumbel_key_func,
        vocab_size=vocab_size,
        dist=dist8,
        null=null,
        normalize=False
    )
    test_stats.append(test_stat8)

    def dist9(x, y): return ems_yes_adaptive_score(
        x, y, torch.from_numpy(genfromtxt(args.token_file + '-null-empty-probs.csv',
                                          delimiter=',')[Tindex, :])
    )

    def test_stat9(tokens, n, k, generator, vocab_size, null=False): return phi(
        tokens=tokens,
        n=n,
        k=k,
        generator=generator,
        key_func=gumbel_key_func,
        vocab_size=vocab_size,
        dist=dist9,
        null=null,
        normalize=False
    )
    test_stats.append(test_stat9)

else:
    raise

ds_iterator = iter(dataset)


def test(tokens, seed, test_stats): return permutation_test(tokens,
                                                            vocab_size,
                                                            n,
                                                            k,
                                                            seed,
                                                            test_stats,
                                                            n_runs=args.n_runs)


t1 = time.time()

csv_saves = []
csvWriters = []
if args.method == "transform":
    csv_saves.append(open(args.token_file + '/' +
                     str(args.Tindex) + '-transform-edit.csv', 'w'))
    csvWriters.append(csv.writer(csv_saves[-1], delimiter=','))
    csv_saves.append(open(args.token_file + '/' +
                     str(args.Tindex) + '-transform.csv', 'w'))
    csvWriters.append(csv.writer(csv_saves[-1], delimiter=','))
    csv_saves.append(open(args.token_file + '/' +
                     str(args.Tindex) + '-its.csv', 'w'))
    csvWriters.append(csv.writer(csv_saves[-1], delimiter=','))
    csv_saves.append(open(args.token_file + '/' +
                          str(args.Tindex) + '-itsl.csv', 'w'))
    csvWriters.append(csv.writer(csv_saves[-1], delimiter=','))
elif args.method == "gumbel":
    # csv_saves.append(open(args.token_file + '-detect/watermarked-' +
    #                  str(args.Tindex) + '-gumbel-edit.csv', 'w'))
    # csvWriters.append(csv.writer(csv_saves[-1], delimiter=','))
    # csv_saves.append(open(args.token_file + '-detect/watermarked-' +
    #                  str(args.Tindex) + '-gumbel.csv', 'w'))
    # csvWriters.append(csv.writer(csv_saves[-1], delimiter=','))
    # csv_saves.append(open(args.token_file + '-detect/watermarked-' +
    #                  str(args.Tindex) + '-ems.csv', 'w'))
    # csvWriters.append(csv.writer(csv_saves[-1], delimiter=','))
    # csv_saves.append(open(args.token_file + '-detect/watermarked-' +
    #                  str(args.Tindex) + '-emsl.csv', 'w'))
    # csvWriters.append(csv.writer(csv_saves[-1], delimiter=','))
    csv_saves.append(open(args.token_file + '-detect/watermarked-' +
                     str(args.Tindex) + '.csv', 'w'))
    csvWriters.append(csv.writer(csv_saves[-1], delimiter=','))
    # csv_saves.append(open(args.token_file + '-detect/null-' +
    #                  str(args.Tindex) + '-gumbel-edit.csv', 'w'))
    # csvWriters.append(csv.writer(csv_saves[-1], delimiter=','))
    # csv_saves.append(open(args.token_file + '-detect/null-' +
    #                  str(args.Tindex) + '-gumbel.csv', 'w'))
    # csvWriters.append(csv.writer(csv_saves[-1], delimiter=','))
    # csv_saves.append(open(args.token_file + '-detect/null-' +
    #                  str(args.Tindex) + '-ems.csv', 'w'))
    # csvWriters.append(csv.writer(csv_saves[-1], delimiter=','))
    # csv_saves.append(open(args.token_file + '-detect/null-' +
    #                  str(args.Tindex) + '-emsl.csv', 'w'))
    # csvWriters.append(csv.writer(csv_saves[-1], delimiter=','))
    # csv_saves.append(open(args.token_file + '-detect/null-' +
    #                  str(args.Tindex) + '-ems-not-adaptive.csv', 'w'))
    # csvWriters.append(csv.writer(csv_saves[-1], delimiter=','))
    # csv_saves.append(open(args.token_file + '-detect/null-' +
    #                  str(args.Tindex) + '-ems-yes-adaptive.csv', 'w'))
    # csvWriters.append(csv.writer(csv_saves[-1], delimiter=','))
    # csv_saves.append(open(args.token_file + '-detect/null-' +
    #                  str(args.Tindex) + '-ems-yes-adaptive-empty.csv', 'w'))
    # csvWriters.append(csv.writer(csv_saves[-1], delimiter=','))
else:
    raise

watermarked_sample = watermarked_samples[Tindex, :]
null_sample = null_samples[Tindex, :]

t0 = time.time()
watermarked_pval = test(watermarked_sample, seeds[Tindex], [
                        test_stats[i] for i in [0, 1, 2]])
log_file.write(f'Ran watermarked test in (t = {time.time()-t0} seconds)\n')
log_file.flush()
# t0 = time.time()
# null_pval = test(null_sample, seeds[Tindex], [
#                  test_stats[i] for i in [0, 3, 4,]])
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
