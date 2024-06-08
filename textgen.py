from time import time

import torch

from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import MarianMTModel, MarianTokenizer

from datasets import load_dataset

from tqdm import tqdm
from collections import defaultdict
import pickle
import copy

import numpy as np

from watermarking.generation import generate, generate_rnd
from watermarking.attacks import deletion_attack, insertion_attack, substitution_attack

from watermarking.transform.sampler import transform_sampling
from watermarking.transform.key import transform_key_func

from watermarking.gumbel.sampler import gumbel_sampling
from watermarking.gumbel.key import gumbel_key_func

import argparse

import csv

results = defaultdict(dict)

parser = argparse.ArgumentParser(description="Experiment Settings")

parser.add_argument('--method', default="transform", type=str)

parser.add_argument('--model', default="facebook/opt-1.3b", type=str)
parser.add_argument('--save', default="", type=str)
parser.add_argument('--seed', default=0, type=int)
parser.add_argument('--batch_size', default=1, type=int)

parser.add_argument('--m', default=80, type=int)
parser.add_argument('--k', default=0, type=int)
parser.add_argument('--n', default=256, type=int)
parser.add_argument('--T', default=500, type=int)

parser.add_argument('--prompt_tokens', default=50, type=int)
parser.add_argument('--buffer_tokens', default=20, type=int)
parser.add_argument('--n_runs', default=5000, type=int)
parser.add_argument('--max_seed', default=100000, type=int)
parser.add_argument('--offset', action='store_true')

parser.add_argument('--gamma', default=0.4, type=float)

parser.add_argument('--deletion', default=0.0, type=float)
parser.add_argument('--insertion', default=0.0, type=float)
parser.add_argument('--substitution', default=0.0, type=float)

parser.add_argument('--kirch_gamma', default=0.25, type=float)
parser.add_argument('--kirch_delta', default=1.0, type=float)

parser.add_argument('--rt_translate', action='store_true')
parser.add_argument('--language', default="french", type=str)

parser.add_argument('--truncate_vocab', default=8, type=int)

args = parser.parse_args()
results['args'] = copy.deepcopy(args)

log_file = open('log/textgen.log', 'w')
log_file.write(str(args) + '\n')
log_file.flush()

# fix the random seed for reproducibility
t0 = time()
torch.manual_seed(args.seed)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained(args.model)
model = AutoModelForCausalLM.from_pretrained(args.model).to(device)

vocab_size = model.get_output_embeddings().weight.shape[0]
eff_vocab_size = vocab_size - args.truncate_vocab
log_file.write(f'Loaded the model (t = {time()-t0} seconds)\n')
log_file.flush()

dataset = load_dataset("allenai/c4", "realnewslike",
                       split="train", streaming=True)


def corrupt(tokens):
    tokens = deletion_attack(tokens, args.deletion)
    tokens = insertion_attack(tokens, args.insertion, eff_vocab_size)
    tokens = substitution_attack(tokens, args.substitution, eff_vocab_size)

    return tokens


T = args.T                  # number of prompts/generations
n_batches = int(np.ceil(T / args.batch_size))  # number of batches
prompt_tokens = args.prompt_tokens      # minimum prompt length
new_tokens = args.m     # number of tokens to generate
buffer_tokens = args.buffer_tokens
if args.k == 0:
    k = args.m  # k is the block size (= number of tokens)
else:
    k = args.k
n = args.n     # watermark key length

if args.rt_translate:
    if args.language == "french":
        en_ne_model_name = "Helsinki-NLP/opus-mt-tc-big-en-fr"
        en_ne_tokenizer = MarianTokenizer.from_pretrained(en_ne_model_name)
        en_ne_model = MarianMTModel.from_pretrained(
            en_ne_model_name).to(device)

        ne_en_model_name = "Helsinki-NLP/opus-mt-tc-big-fr-en"
        ne_en_tokenizer = MarianTokenizer.from_pretrained(ne_en_model_name)
        ne_en_model = MarianMTModel.from_pretrained(
            ne_en_model_name).to(device)
    elif args.language == "russian":
        en_ne_model_name = "Helsinki-NLP/opus-mt-en-ru"
        en_ne_tokenizer = MarianTokenizer.from_pretrained(en_ne_model_name)
        en_ne_model = MarianMTModel.from_pretrained(
            en_ne_model_name).to(device)

        ne_en_model_name = "Helsinki-NLP/opus-mt-ru-en"
        ne_en_tokenizer = MarianTokenizer.from_pretrained(ne_en_model_name)
        ne_en_model = MarianMTModel.from_pretrained(
            ne_en_model_name).to(device)
    else:
        raise

    def rt_translate(text):
        try:
            tokens = en_ne_tokenizer(text.split(
                '. '), return_tensors="pt", padding=True).to(device)
            tokens = en_ne_model.generate(**tokens, max_new_tokens=52)
            french_text = ' '.join([en_ne_tokenizer.decode(
                t, skip_special_tokens=True) for t in tokens])

            tokens = ne_en_tokenizer(french_text.split(
                '. '), return_tensors="pt", padding=True).to(device)
            tokens = ne_en_model.generate(**tokens, max_new_tokens=512)
            roundtrip_text = ' '.join([ne_en_tokenizer.decode(
                t, skip_special_tokens=True) for t in tokens])
        except:
            roundtrip_text = ""
        return roundtrip_text

# this is the "key" for the watermark
# for now each generation gets its own key
seeds = torch.randint(2**32, (T,))
seeds_save = open(args.save + '-seeds.csv', 'w')
seeds_writer = csv.writer(seeds_save, delimiter=",")
seeds_writer.writerow(np.asarray(seeds.squeeze().numpy()))
seeds_save.close()

if args.method == "transform":
    def generate_watermark(prompt, seed, empty_prompts): return generate(
        model,
        prompt,
        vocab_size,
        n,
        new_tokens+buffer_tokens,
        seed,
        transform_key_func,
        transform_sampling,
        random_offset=args.offset,
        empty_prompts=empty_prompts
    )

elif args.method == "gumbel":
    def generate_watermark(prompt, seed, empty_prompts): return generate(
        model,
        prompt,
        vocab_size,
        n,
        new_tokens+buffer_tokens,
        seed,
        gumbel_key_func,
        gumbel_sampling,
        random_offset=args.offset,
        empty_prompts=empty_prompts
    )
else:
    raise

ds_iterator = iter(dataset)

t1 = time()

prompt_save = open(args.save + '-prompt.csv', 'w')
prompt_writer = csv.writer(prompt_save, delimiter=",")
prompts = []
itm = 0
pbar = tqdm(total=T)
while itm < T:
    example = next(ds_iterator)
    text = example['text']

    tokens = tokenizer.encode(
        text,
        return_tensors='pt',
        truncation=True,
        max_length=2048-buffer_tokens
    )[0]
    if len(tokens) < prompt_tokens + new_tokens:
        continue
    prompt = tokens[-(new_tokens+prompt_tokens):-new_tokens]
    prompts.append(prompt)
    prompt_writer.writerow(np.asarray(prompt.numpy()))

    itm += 1
    pbar.update(1)

empty_prompt_save = open(args.save + '-empty-prompt.txt', 'w')
if args.model == "facebook/opt-1.3b":
    empty_text = ""
elif args.model == "openai-community/gpt2":
    empty_text = " "
else:
    raise
empty_token = tokenizer.encode(
    empty_text,
    return_tensors='pt',
    truncation=True,
    max_length=2048-buffer_tokens
)[0]
empty_prompts = [empty_token for _ in range(T)]
empty_prompt_save.write(str(empty_token))
empty_prompt_save.close()

pbar.close()
prompt_save.close()

prompts = torch.vstack(prompts)
empty_prompts = torch.vstack(empty_prompts)

null_samples = []
watermarked_samples = []

null_probs = []
watermarked_probs = []

null_empty_probs = []
watermarked_empty_probs = []

pbar = tqdm(total=n_batches)
for batch in range(n_batches):
    idx = torch.arange(batch * args.batch_size,
                       min(T, (batch + 1) * args.batch_size))

    null_sample, null_prob, null_empty_prob = generate_rnd(
        prompts[idx], new_tokens+buffer_tokens, model, empty_prompts[idx])
    null_samples.append(null_sample[:, (prompt_tokens + buffer_tokens):])
    null_probs.append(null_prob[:, buffer_tokens:])
    null_empty_probs.append(null_empty_prob[:, buffer_tokens:])
    watermarked_sample, watermarked_prob, watermarked_empty_prob = generate_watermark(
        prompts[idx], seeds[idx], empty_prompts[idx])
    watermarked_samples.append(watermarked_sample[:, (prompt_tokens + buffer_tokens):])
    watermarked_probs.append(watermarked_prob[:, buffer_tokens:])
    watermarked_empty_probs.append(watermarked_empty_prob[:, buffer_tokens:])
    pbar.update(1)
pbar.close()
null_samples = torch.vstack(null_samples)
watermarked_samples = torch.vstack(watermarked_samples)

null_probs = torch.vstack(null_probs)
null_empty_probs = torch.vstack(null_empty_probs)

watermarked_probs = torch.vstack(watermarked_probs)
watermarked_empty_probs = torch.vstack(watermarked_empty_probs)

results['watermark']['tokens'] = copy.deepcopy(watermarked_samples)
results['null']['tokens'] = copy.deepcopy(null_samples)

null_samples = torch.clip(null_samples, max=eff_vocab_size-1)
watermarked_samples = torch.clip(watermarked_samples, max=eff_vocab_size-1)

log_file.write(f'Generated samples in (t = {time()-t1} seconds)\n')
log_file.flush()

t1 = time()
tokens_before_attack_save = open(args.save + '-tokens-before-attack.csv', "w")
probs_save = open(args.save + '-probs.csv', "w")
empty_probs_save = open(args.save + '-empty-probs.csv', "w")
tokens_before_attack_writer = csv.writer(
    tokens_before_attack_save, delimiter=",")
probs_writer = csv.writer(probs_save, delimiter=",")
empty_probs_writer = csv.writer(empty_probs_save, delimiter=",")
pbar = tqdm(total=len(watermarked_samples))
for tokens, probs, empty_probs in zip(watermarked_samples, watermarked_probs, watermarked_empty_probs):
    tokens_before_attack_writer.writerow(np.asarray(tokens.numpy()))
    probs_writer.writerow(np.asarray(probs.numpy()))
    empty_probs_writer.writerow(np.asarray(empty_probs.numpy()))
    pbar.update(1)
pbar.close()
tokens_before_attack_save.close()
probs_save.close()
empty_probs_save.close()
log_file.write(
    f'Saved text/tokens before attack and probs in (t = {time()-t1} seconds)\n')
log_file.flush()

t1 = time()
null_tokens_save = open(args.save + '-null.csv', 'w')
null_probs_save = open(args.save + '-null-probs.csv', 'w')
null_empty_probs_save = open(args.save + '-null-empty-probs.csv', 'w')
null_tokens_writer = csv.writer(null_tokens_save, delimiter=",")
null_probs_writer = csv.writer(null_probs_save, delimiter=",")
null_empty_probs_writer = csv.writer(null_empty_probs_save, delimiter=",")
pbar = tqdm(total=len(null_samples))
for tokens, probs, empty_probs in zip(null_samples, null_probs, null_empty_probs):
    null_tokens_writer.writerow(np.asarray(tokens.numpy()))
    null_probs_writer.writerow(np.asarray(probs.numpy()))
    null_empty_probs_writer.writerow(np.asarray(empty_probs.numpy()))
    pbar.update(1)
pbar.close()
null_tokens_save.close()
null_probs_save.close()
null_empty_probs_save.close()
log_file.write(
    f'Saved null samples and probs in (t = {time()-t1} seconds)\n')
log_file.flush()

attacked_tokens_save = open(
    args.save + "-attacked-tokens.csv", "w")
attacked_tokens_writer = csv.writer(attacked_tokens_save, delimiter=",")
pi_save = None
pi_writer = None
if args.method == "transform":
    pi_save = open(args.save + "-pi.csv", "w")
    pi_writer = csv.writer(pi_save, delimiter=",")

pbar = tqdm(total=T)
for itm in range(T):
    watermarked_sample = watermarked_samples[itm]
    watermarked_sample = corrupt(watermarked_sample)
    watermarked_sample = tokenizer.decode(
        watermarked_sample, skip_special_tokens=True)
    if args.rt_translate:
        watermarked_sample = rt_translate(watermarked_sample)
    watermarked_sample = tokenizer.encode(watermarked_sample,
                                          return_tensors='pt',
                                          truncation=True,
                                          max_length=2048)[0]
    if len(watermarked_sample) < new_tokens + 1:
        watermarked_sample = torch.nn.functional.pad(
            watermarked_sample, (new_tokens-len(watermarked_sample), 0),
            "constant", 0
        )
    else:
        watermarked_sample = watermarked_sample[1:new_tokens+1]
    attacked_tokens_writer.writerow(np.asarray(watermarked_sample.numpy()))
    if args.method == "transform":
        generator = torch.Generator()
        generator.manual_seed(int(seeds[itm]))
        pi = torch.randperm(vocab_size, generator=generator)
        pi_writer.writerow(np.asarray(pi.squeeze().numpy()))
    elif args.method == "gumbel":
        pass
    else:
        raise

    pbar.update(1)

pbar.close()
log_file.write(f'Attacked the samples in (t = {time()-t1} seconds)\n')
log_file.flush()
log_file.close()
attacked_tokens_save.close()

pickle.dump(results, open(args.save, "wb"))
