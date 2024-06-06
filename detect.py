import time
import torch
from datasets import load_dataset

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
parser.add_argument('--nowatermark', action='store_true')

parser.add_argument('--deletion', default=0.0, type=float)
parser.add_argument('--insertion', default=0.0, type=float)
parser.add_argument('--substitution', default=0.0, type=float)

parser.add_argument('--kirch_gamma', default=0.25, type=float)
parser.add_argument('--kirch_delta', default=1.0, type=float)

parser.add_argument('--truncate_vocab', default=8, type=int)

args = parser.parse_args()
results['args'] = copy.deepcopy(args)

log_file = open(
    'log/' + str(args.Tindex) + "-" + str(args.deletion) + "-" +
    str(args.insertion) + "-" + str(args.substitution) + '.log', 'w'
)
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

    def dist0(x, y): return ems_score(x, y)

    def test_stat0(tokens, n, k, generator, vocab_size, null=False): return phi(
        tokens=tokens,
        n=n,
        k=k,
        generator=generator,
        key_func=gumbel_key_func,
        vocab_size=vocab_size,
        dist=dist0,
        null=null,
        normalize=False
    )
    test_stats.append(test_stat0)

    def dist1(x, y): return ems_adaptive(
        x, y, torch.from_numpy(genfromtxt(
            args.token_file + '-empty-probs.csv', delimiter=','
        )[Tindex, :]), 1.0
    )

    def test_stat1(tokens, n, k, generator, vocab_size, null=False): return phi(
        tokens=tokens,
        n=n,
        k=k,
        generator=generator,
        key_func=gumbel_key_func,
        vocab_size=vocab_size,
        dist=dist1,
        null=null,
        normalize=False
    )
    test_stats.append(test_stat1)

    def dist2(x, y): return ems_adaptive(
        x, y, torch.from_numpy(genfromtxt(
            args.token_file + '-empty-probs.csv', delimiter=','
        )[Tindex, :]), 0.75
    )

    def test_stat2(tokens, n, k, generator, vocab_size, null=False): return phi(
        tokens=tokens,
        n=n,
        k=k,
        generator=generator,
        key_func=gumbel_key_func,
        vocab_size=vocab_size,
        dist=dist2,
        null=null,
        normalize=False
    )
    test_stats.append(test_stat2)

    def dist3(x, y): return ems_adaptive(
        x, y, torch.from_numpy(genfromtxt(
            args.token_file + '-empty-probs.csv', delimiter=','
        )[Tindex, :]), 0.5
    )

    def test_stat3(tokens, n, k, generator, vocab_size, null=False): return phi(
        tokens=tokens,
        n=n,
        k=k,
        generator=generator,
        key_func=gumbel_key_func,
        vocab_size=vocab_size,
        dist=dist3,
        null=null,
        normalize=False
    )
    test_stats.append(test_stat3)

    def dist4(x, y): return ems_adaptive(
        x, y, torch.from_numpy(genfromtxt(
            args.token_file + '-empty-probs.csv', delimiter=','
        )[Tindex, :]), 0.25
    )

    def test_stat4(tokens, n, k, generator, vocab_size, null=False): return phi(
        tokens=tokens,
        n=n,
        k=k,
        generator=generator,
        key_func=gumbel_key_func,
        vocab_size=vocab_size,
        dist=dist4,
        null=null,
        normalize=False
    )
    test_stats.append(test_stat4)

    # def dist2(x, y): return ems_adaptive(
    #     x, y, torch.from_numpy(genfromtxt(
    #         args.token_file + '-null-empty-probs.csv', delimiter=','
    #     )[Tindex, :])
    # )

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

else:
    raise

ds_iterator = iter(dataset)


def test(tokens, seed, test_stats): return permutation_test(tokens,
                                                            vocab_size,
                                                            n,
                                                            k,
                                                            seed,
                                                            test_stats,
                                                            log_file,
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
    csv_saves.append(open(args.token_file + '-detect/watermarked-' +
                     str(args.Tindex) + '.csv', 'w'))
    csvWriters.append(csv.writer(csv_saves[-1], delimiter=','))
    # csv_saves.append(open(args.token_file + '-detect/null-' +
    #                  str(args.Tindex) + '.csv', 'w'))
    # csvWriters.append(csv.writer(csv_saves[-1], delimiter=','))
else:
    raise

watermarked_sample = watermarked_samples[Tindex, :]
null_sample = null_samples[Tindex, :]

t0 = time.time()
watermarked_pval = test(watermarked_sample, seeds[Tindex], [
                        test_stats[i] for i in [0, 1, 2, 3, 4]])
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
