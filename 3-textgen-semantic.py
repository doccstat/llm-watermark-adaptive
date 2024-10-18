from time import time

import torch

from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import MarianMTModel, MarianTokenizer

from datasets import load_dataset, load_from_disk

from tqdm import tqdm
import copy

import numpy as np

from watermarking.generation import generate, generate_rnd, generate_mixed
from watermarking.attacks import deletion_attack_semantic, insertion_attack_semantic

from watermarking.transform.sampler import transform_sampling
from watermarking.transform.key import transform_key_func

from watermarking.gumbel.sampler import gumbel_sampling
from watermarking.gumbel.key import gumbel_key_func

import argparse

import csv
import json

parser = argparse.ArgumentParser(description="Experiment Settings")

parser.add_argument('--method', default="transform", type=str)

parser.add_argument('--model', default="facebook/opt-1.3b", type=str)
parser.add_argument('--save', default="", type=str)
parser.add_argument('--seed', default=0, type=int)
parser.add_argument('--batch_size', default=1, type=int)

parser.add_argument('--tokens_count', default=80, type=int)
parser.add_argument('--k', default=0, type=int)
parser.add_argument('--watermark_key_length', default=256, type=int)
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

log_file = open('log/textgen.log', 'w')
log_file.write(str(args) + '\n')
log_file.flush()

# fix the random seed for reproducibility
t0 = time()
torch.manual_seed(args.seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

try:
    tokenizer = AutoTokenizer.from_pretrained(
        "/scratch/user/anthony.li/models/" + args.model + "/tokenizer")
    model = AutoModelForCausalLM.from_pretrained(
        "/scratch/user/anthony.li/models/" + args.model + "/model",
        device_map='auto'
    )

    log_file.write(f'Loaded the local model\n')
except:
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model).to(device)
    log_file.write(f'Loaded the model\n')

log_file.flush()

vocab_size = model.get_output_embeddings().weight.shape[0]
eff_vocab_size = vocab_size - args.truncate_vocab
log_file.write(f'Loaded the model (t = {time()-t0} seconds)\n')
log_file.flush()

try:
    dataset = load_from_disk(
        '/scratch/user/anthony.li/datasets/allenai/c4/realnewslike/train'
    )
except:
    dataset = load_dataset("allenai/c4", "realnewslike",
                           split="train", streaming=True)


T = args.T                  # number of prompts/generations
n_batches = int(np.ceil(T / args.batch_size))  # number of batches
prompt_tokens = args.prompt_tokens      # minimum prompt length
new_tokens = args.tokens_count

# Generate more tokens if we are going to delete some.
if args.deletion:
    new_tokens += 20
buffer_tokens = args.buffer_tokens
if args.k == 0:
    k = args.tokens_count  # k is the block size (= number of tokens)
else:
    k = args.k
n = args.watermark_key_length

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
    def generate_watermark(prompt, seed, empty_prompts, fixed_inputs=None):
        return generate(
            model,
            prompt,
            vocab_size,
            n,
            new_tokens+buffer_tokens,
            seed,
            transform_key_func,
            transform_sampling,
            random_offset=args.offset,
            empty_prompts=empty_prompts,
            fixed_inputs=fixed_inputs
        )

    def generate_watermark_mixed(prompt, seed, empty_prompts, no_watermark_locations):
        return generate_mixed(
            model,
            prompt,
            vocab_size,
            n,
            new_tokens+buffer_tokens,
            seed,
            transform_key_func,
            transform_sampling,
            random_offset=args.offset,
            empty_prompts=empty_prompts,
            fixed_inputs=None,
            no_watermark_locations=no_watermark_locations
        )

elif args.method == "gumbel":
    def generate_watermark(prompt, seed, empty_prompts, fixed_inputs=None):
        return generate(
            model,
            prompt,
            vocab_size,
            n,
            new_tokens+buffer_tokens,
            seed,
            gumbel_key_func,
            gumbel_sampling,
            random_offset=args.offset,
            empty_prompts=empty_prompts,
            fixed_inputs=fixed_inputs
        )

    def generate_watermark_mixed(prompt, seed, empty_prompts, no_watermark_locations):
        return generate_mixed(
            model,
            prompt,
            vocab_size,
            n,
            new_tokens+buffer_tokens,
            seed,
            gumbel_key_func,
            gumbel_sampling,
            random_offset=args.offset,
            empty_prompts=empty_prompts,
            fixed_inputs=None,
            no_watermark_locations=no_watermark_locations
        )
else:
    raise

ds_iterator = iter(dataset)

t1 = time()

# Iterate through the dataset to get the prompts
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
pbar.close()
prompt_save.close()
prompts = torch.vstack(prompts)

# Generate the candidate prompts that will be used to find the best suited
# prompt for the attacked watermarked texts.
candidate_prompts = []
prompt_copy = copy.deepcopy(prompts)
for i in range(T):
    idx = torch.randint(0, prompt_tokens, (int(0.1*prompt_tokens),))
    prompt_copy[i, idx] = 0
    candidate_prompts.append(
        torch.vstack([prompt_copy[i] for _ in range(T)])
    )

empty_prompt_save = open(args.save + '-empty-prompt.txt', 'w')
if args.model == "facebook/opt-1.3b":
    candidate_prompt = ""
elif args.model == "openai-community/gpt2":
    candidate_prompt = " "
elif args.model == "meta-llama/Meta-Llama-3-8B":
    candidate_prompt = ""
elif args.model == "mistralai/Mistral-7B-v0.1":
    candidate_prompt = ""
else:
    raise
candidate_token = tokenizer.encode(
    candidate_prompt,
    return_tensors='pt',
    truncation=True,
    max_length=2048 - buffer_tokens
)[0]
empty_prompt_save.write(str(candidate_token))
empty_prompt_save.close()

# The last candidate prompt is the empty prompt. Later in the script another
# set of prompts will be appended generated by the model itself based on the
# attacked watermarked texts.
candidate_prompts.append(torch.vstack([candidate_token for _ in range(T)]))
watermarked_samples, watermarked_probs, watermarked_empty_probs = [], [], []
watermarked_ntps, watermarked_empty_ntps = [], []

attacked_idx_save = open(args.save + "-attacked-idx.csv", "w")
attacked_idx_writer = csv.writer(attacked_idx_save, delimiter=",")

pbar = tqdm(total=n_batches)
for batch in range(n_batches):
    idx = torch.arange(batch * args.batch_size,
                       min(T, (batch + 1) * args.batch_size))

    no_watermark_locations = []
    no_watermark_locations_count = int(args.substitution * new_tokens)
    for i in idx:
        no_watermark_locations_start = torch.randint(
            0, new_tokens - no_watermark_locations_count, (1,)).item()
        no_watermark_locations_end = no_watermark_locations_start + \
            no_watermark_locations_count
        no_watermark_locations.append(
            # torch.randperm(new_tokens)[:no_watermark_locations_count]
            torch.arange(no_watermark_locations_start,
                         no_watermark_locations_end)
        )
        attacked_idx_writer.writerow(np.asarray(
            no_watermark_locations[-1].numpy()))
        attacked_idx_save.flush()
    no_watermark_locations = torch.vstack(no_watermark_locations)

    watermarked_sample, watermarked_prob, watermarked_empty_prob, watermarked_ntp, watermarked_empty_ntp = generate_watermark_mixed(
        prompts[idx], seeds[idx], candidate_prompts[-1][idx], no_watermark_locations)
    watermarked_samples.append(watermarked_sample[:, prompt_tokens:])
    watermarked_probs.append(watermarked_prob)
    watermarked_empty_probs.append(watermarked_empty_prob)
    if args.method == "transform":
        watermarked_ntps.append(watermarked_ntp)
        watermarked_empty_ntps.append(watermarked_empty_ntp.cpu())
    pbar.update(1)
pbar.close()
attacked_idx_save.close()

watermarked_samples = torch.vstack(watermarked_samples)
watermarked_probs = torch.vstack(watermarked_probs)
watermarked_empty_probs = torch.vstack(watermarked_empty_probs)

watermarked_samples = torch.clip(watermarked_samples, max=eff_vocab_size-1)

log_file.write(f'Generated samples in (t = {time()-t1} seconds)\n')
log_file.flush()

# Save the text/tokens before attack and NTP for each token in the watermark
# texts with true and empty prompt.
t1 = time()
tokens_before_attack_save = open(args.save + '-tokens-before-attack.csv', "w")
_probs_save = open(args.save + '-probs.csv', "w")
_empty_probs_save = open(args.save + '-empty-probs.csv', "w")
tokens_before_attack_writer = csv.writer(
    tokens_before_attack_save, delimiter=",")
_probs_writer = csv.writer(_probs_save, delimiter=",")
_empty_probs_writer = csv.writer(_empty_probs_save, delimiter=",")
pbar = tqdm(total=len(watermarked_samples))
for tokens, _probs, _empty_probs in zip(
    watermarked_samples, watermarked_probs, watermarked_empty_probs
):
    tokens_before_attack_writer.writerow(np.asarray(tokens.numpy()))
    _probs_writer.writerow(np.asarray(_probs.numpy()[:args.tokens_count]))
    _empty_probs_writer.writerow(np.asarray(
        _empty_probs.numpy()[:args.tokens_count]))
    pbar.update(1)
pbar.close()
tokens_before_attack_save.close()
_probs_save.close()
_empty_probs_save.close()
np.save(args.save + '-ntps.npy', watermarked_ntps)
np.save(args.save + '-empty-ntps.npy', watermarked_empty_ntps)
log_file.write(
    f'Saved text/tokens before attack and probs in (t = {time()-t1} seconds)\n')
log_file.flush()

# Attack the watermarked texts and store a copy appended with the
# prompt-extracting prompt in `icl_samples`.
attacked_tokens_save = open(
    args.save + "-attacked-tokens.csv", "w")
attacked_tokens_writer = csv.writer(attacked_tokens_save, delimiter=",")
pi_save = None
pi_writer = None
if args.method == "transform":
    pi_save = open(args.save + "-pi.csv", "w")
    pi_writer = csv.writer(pi_save, delimiter=",")

attacked_samples = copy.deepcopy(watermarked_samples)
icl_samples = []
icl_sample_max_length = 0

if args.deletion or args.insertion:
    attacked_idx_save = open(args.save + "-attacked-idx.csv", "w")
    attacked_idx_writer = csv.writer(attacked_idx_save, delimiter=",")

pbar = tqdm(total=T)
for itm in range(T):
    watermarked_sample = watermarked_samples[itm]
    if args.deletion:
        watermarked_sample, attack_span = deletion_attack_semantic(
            watermarked_sample, tokenizer)
        if attack_span[0] is None or attack_span[1] is None:
            attacked_idx_writer.writerow(
                np.asarray(torch.arange(0, 0).numpy()))
        else:
            attacked_idx_writer.writerow(np.asarray(
                torch.arange(attack_span[0], attack_span[1]).numpy()))
        attacked_idx_save.flush()
    elif args.insertion:
        watermarked_sample, attack_span = insertion_attack_semantic(
            watermarked_sample,
            prompts[itm],
            tokenizer,
            model,
            max_insert_length=50
        )
        watermarked_sample = watermarked_sample[:new_tokens]
        if attack_span[0] is None or attack_span[1] is None:
            attacked_idx_writer.writerow(
                np.asarray(torch.arange(0, 0).numpy()))
        else:
            attacked_idx_writer.writerow(np.asarray(torch.arange(
                attack_span[0], min(attack_span[1], new_tokens)).numpy()))
        attacked_idx_save.flush()

    watermarked_sample = tokenizer.decode(
        watermarked_sample, skip_special_tokens=True)
    if args.rt_translate:
        watermarked_sample = rt_translate(watermarked_sample)
    icl_samples.append(tokenizer.encode(watermarked_sample + ". What might be the prompt that generated previous text? Start with the prompt directly. ",
                                        return_tensors='pt',
                                        truncation=True,
                                        max_length=2048)[0])
    icl_sample_max_length = max(icl_sample_max_length, len(icl_samples[-1]))
    watermarked_sample = tokenizer.encode(watermarked_sample,
                                          return_tensors='pt',
                                          truncation=True,
                                          max_length=2048)[0]

    # This is very very ad-hoc, but it seems that the tokenizer sometimes adds a special token at the beginning
    # of the sequence. If the first token is 1 or 128000, remove it.
    # 128000 for meta-llama/Meta-Llama-3-8B and 1 for mistralai/Mistral-7B-v0.1
    if (args.model == "meta-llama/Meta-Llama-3-8B" and watermarked_sample[0] == 128000) or (args.model == "mistralai/Mistral-7B-v0.1" and watermarked_sample[0] == 1):
        watermarked_sample = watermarked_sample[1:]
    if len(watermarked_sample) < new_tokens + 1:
        watermarked_sample = torch.nn.functional.pad(
            watermarked_sample, (0, new_tokens-len(watermarked_sample)),
            "constant", 0
        )
    else:
        watermarked_sample = watermarked_sample[1:new_tokens+1]
    attacked_samples[itm] = watermarked_sample
    attacked_tokens_writer.writerow(np.asarray(
        watermarked_sample.numpy()[:args.tokens_count]))
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
if args.deletion or args.insertion:
    attacked_idx_save.close()
attacked_tokens_save.close()

# Pad the icl samples to the maximum length.
icl_samples = [
    torch.nn.functional.pad(
        icl_sample, (0, icl_sample_max_length - len(icl_sample)),
        "constant", candidate_token.item()
    ) for icl_sample in icl_samples
]

# Generate the ICL prompts for the attacked watermarked texts.
icl_prompts = []
pbar = tqdm(total=n_batches)
for batch in range(n_batches):
    idx = torch.arange(batch * args.batch_size,
                       min(T, (batch + 1) * args.batch_size))

    null_sample, _, _ = generate_rnd(
        torch.vstack(icl_samples)[idx], prompt_tokens + buffer_tokens,
        model, candidate_prompts[-1][idx]
    )
    icl_prompts.append(null_sample[:, icl_sample_max_length:])

    pbar.update(1)
pbar.close()
icl_prompts = torch.vstack(icl_prompts)
candidate_prompts.append(icl_prompts)

icl_prompt_save = open(args.save + '-icl-prompt.csv', 'w')
icl_prompt_writer = csv.writer(icl_prompt_save, delimiter=",")
for icl_prompt in icl_prompts:
    icl_prompt_writer.writerow(np.asarray(icl_prompt.numpy()))
icl_prompt_save.close()

re_calculated_probs = []
re_calculated_best_probs = []
re_calculated_empty_probs = []
re_calculated_icl_probs = []
best_prompt = []

re_calculated_probs_save = open(
    args.save + "-re-calculated-probs.csv", "w")
re_calculated_probs_writer = csv.writer(
    re_calculated_probs_save, delimiter=",")
re_calculated_best_probs_save = open(
    args.save + "-re-calculated-best-probs.csv", "w")
re_calculated_best_probs_writer = csv.writer(
    re_calculated_best_probs_save, delimiter=",")
re_calculated_empty_probs_save = open(
    args.save + "-re-calculated-empty-probs.csv", "w")
re_calculated_empty_probs_writer = csv.writer(
    re_calculated_empty_probs_save, delimiter=",")
re_calculated_icl_probs_save = open(
    args.save + "-re-calculated-icl-probs.csv", "w")
re_calculated_icl_probs_writer = csv.writer(
    re_calculated_icl_probs_save, delimiter=",")
best_prompt_save = open(args.save + '-best-prompt.csv', 'w')
best_prompt_writer = csv.writer(best_prompt_save, delimiter=",")
re_calculated_ntps, re_calculated_best_ntps, re_calculated_empty_ntps, re_calculated_icl_ntps = [], [], [], []

pbar = tqdm(total=n_batches * (8 + 2))
for batch in range(n_batches):
    idx = torch.arange(batch * args.batch_size,
                       min(T, (batch + 1) * args.batch_size))

    candidate_prompts_subset = [
        torch.zeros((T, prompt_tokens), dtype=torch.long),
    ]
    for itm in range(T):
        candidate_prompts_subset[0][itm] = candidate_prompts[itm][0]
    candidate_prompts_subset.extend([
        candidate_prompts[itm] for itm in idx.tolist()[:(8 - 1)]
    ])
    candidate_prompts_subset.append(candidate_prompts[-2])
    candidate_prompts_subset.append(candidate_prompts[-1])
    candidate_probs, candidate_empty_ntps = [], []
    for candidate_prompt_idx, candidate_prompt in enumerate(candidate_prompts_subset):
        _, watermarked_prob, watermarked_empty_prob, ntps, empty_ntps = generate_watermark(
            prompts[idx], seeds[idx], candidate_prompt[idx],
            fixed_inputs=attacked_samples[idx])

        # `watermarked_empty_prob` is of shape (len(idx), new_tokens)
        candidate_probs.append(watermarked_empty_prob)

        if args.method == "transform":
            # all probs and all empty probs are of shape (len(idx), vocab_size, new_tokens)
            candidate_empty_ntps.append(empty_ntps.cpu())

        if candidate_prompt_idx == 0:
            re_calculated_probs.append(watermarked_prob)
            if args.method == "transform":
                re_calculated_ntps.append(ntps)
        if candidate_prompt_idx == len(candidate_prompts_subset) - 2:
            re_calculated_empty_probs.append(watermarked_empty_prob)
            if args.method == "transform":
                re_calculated_empty_ntps.append(empty_ntps.cpu())
        elif candidate_prompt_idx == len(candidate_prompts_subset) - 1:
            re_calculated_icl_probs.append(watermarked_empty_prob)
            if args.method == "transform":
                re_calculated_icl_ntps.append(empty_ntps.cpu())

        pbar.update(1)

    # Convert list to tensor before applying tensor operations
    # `candidate_probs` is of shape (len(candidate_prompts_subset), len(idx), new_tokens)
    candidate_probs = torch.stack(candidate_probs)

    if args.method == "transform":
        # `candidate_empty_ntps` are of shape (len(candidate_prompts_subset), len(idx), vocab_size, new_tokens)
        candidate_empty_ntps = torch.stack(candidate_empty_ntps)

    # Now perform the log and sum operations on the tensor
    best_candidate_idx = torch.argmax(
        torch.sum(torch.log(candidate_probs), 2), 0
    )

    re_calculated_best_probs.append(
        candidate_probs[best_candidate_idx, torch.arange(len(idx)), :]
    )
    if args.method == "transform":
        re_calculated_best_ntps.append(
            candidate_empty_ntps[best_candidate_idx,
                                 torch.arange(len(idx)), :, :]
        )

    for bc_idx, itm in zip(best_candidate_idx.tolist(), idx):
        best_prompt.append(candidate_prompts_subset[bc_idx][itm])

best_prompt = [
    torch.nn.functional.pad(
        best_prompt[itm], (0, prompt_tokens - len(best_prompt[itm])),
        "constant", 0
    ) for itm in range(T)
]
best_prompt = torch.vstack(best_prompt)

pbar.close()
re_calculated_probs = torch.vstack(re_calculated_probs)
re_calculated_best_probs = torch.vstack(re_calculated_best_probs)
re_calculated_empty_probs = torch.vstack(re_calculated_empty_probs)
re_calculated_icl_probs = torch.vstack(re_calculated_icl_probs)
for itm in range(T):
    re_calculated_probs_writer.writerow(
        np.asarray(re_calculated_probs[itm].numpy()[:args.tokens_count]))
    re_calculated_best_probs_writer.writerow(
        np.asarray(re_calculated_best_probs[itm].numpy()[:args.tokens_count]))
    re_calculated_empty_probs_writer.writerow(
        np.asarray(re_calculated_empty_probs[itm].numpy()[:args.tokens_count]))
    re_calculated_icl_probs_writer.writerow(
        np.asarray(re_calculated_icl_probs[itm].numpy()[:args.tokens_count]))
    best_prompt_writer.writerow(np.asarray(best_prompt[itm].numpy()))
re_calculated_probs_save.close()
re_calculated_best_probs_save.close()
re_calculated_empty_probs_save.close()
re_calculated_icl_probs_save.close()
best_prompt_save.close()
np.save(args.save + '-re-calculated-ntps.npy', re_calculated_ntps)
np.save(args.save + '-re-calculated-best-ntps.npy', re_calculated_best_ntps)
np.save(args.save + '-re-calculated-empty-ntps.npy', re_calculated_empty_ntps)
np.save(args.save + '-re-calculated-icl-ntps.npy', re_calculated_icl_ntps)

re_calculated_98_probs = []
re_calculated_96_probs = []
re_calculated_90_probs = []
re_calculated_80_probs = []
re_calculated_60_probs = []
re_calculated_40_probs = []
re_calculated_20_probs = []
re_calculated_98_probs_save = open(
    args.save + "-re-calculated-98-probs.csv", "w")
re_calculated_98_probs_writer = csv.writer(
    re_calculated_98_probs_save, delimiter=",")
re_calculated_96_probs_save = open(
    args.save + "-re-calculated-96-probs.csv", "w")
re_calculated_96_probs_writer = csv.writer(
    re_calculated_96_probs_save, delimiter=",")
re_calculated_90_probs_save = open(
    args.save + "-re-calculated-90-probs.csv", "w")
re_calculated_90_probs_writer = csv.writer(
    re_calculated_90_probs_save, delimiter=",")
re_calculated_80_probs_save = open(
    args.save + "-re-calculated-80-probs.csv", "w")
re_calculated_80_probs_writer = csv.writer(
    re_calculated_80_probs_save, delimiter=",")
re_calculated_60_probs_save = open(
    args.save + "-re-calculated-60-probs.csv", "w")
re_calculated_60_probs_writer = csv.writer(
    re_calculated_60_probs_save, delimiter=",")
re_calculated_40_probs_save = open(
    args.save + "-re-calculated-40-probs.csv", "w")
re_calculated_40_probs_writer = csv.writer(
    re_calculated_40_probs_save, delimiter=",")
re_calculated_20_probs_save = open(
    args.save + "-re-calculated-20-probs.csv", "w")
re_calculated_20_probs_writer = csv.writer(
    re_calculated_20_probs_save, delimiter=",")
re_calculated_98_ntps, re_calculated_96_ntps, re_calculated_90_ntps, re_calculated_80_ntps, re_calculated_60_ntps, re_calculated_40_ntps, re_calculated_20_ntps = [], [], [], [], [], [], []

# Create modified prompts for the 20%, 40%, 60%, 80%, 90%, 96%, and 98% cases.
candidate_98_prompts = []
prompt_copy = copy.deepcopy(prompts)
for i in range(T):
    idx = torch.randint(0, prompt_tokens, (int(0.02*prompt_tokens),))
    prompt_copy[i, idx] = 0
    candidate_98_prompts.append(prompt_copy[i])
candidate_98_prompts = torch.vstack(candidate_98_prompts)
candidate_96_prompts = []
prompt_copy = copy.deepcopy(prompts)
for i in range(T):
    idx = torch.randint(0, prompt_tokens, (int(0.04*prompt_tokens),))
    prompt_copy[i, idx] = 0
    candidate_96_prompts.append(prompt_copy[i])
candidate_96_prompts = torch.vstack(candidate_96_prompts)
candidate_90_prompts = []
prompt_copy = copy.deepcopy(prompts)
for i in range(T):
    idx = torch.randint(0, prompt_tokens, (int(0.1*prompt_tokens),))
    prompt_copy[i, idx] = 0
    candidate_90_prompts.append(prompt_copy[i])
candidate_90_prompts = torch.vstack(candidate_90_prompts)
candidate_80_prompts = []
prompt_copy = copy.deepcopy(prompts)
for i in range(T):
    idx = torch.randint(0, prompt_tokens, (int(0.2*prompt_tokens),))
    prompt_copy[i, idx] = 0
    candidate_80_prompts.append(prompt_copy[i])
candidate_80_prompts = torch.vstack(candidate_80_prompts)
candidate_60_prompts = []
prompt_copy = copy.deepcopy(prompts)
for i in range(T):
    idx = torch.randint(0, prompt_tokens, (int(0.4*prompt_tokens),))
    prompt_copy[i, idx] = 0
    candidate_60_prompts.append(prompt_copy[i])
candidate_60_prompts = torch.vstack(candidate_60_prompts)
candidate_40_prompts = []
prompt_copy = copy.deepcopy(prompts)
for i in range(T):
    idx = torch.randint(0, prompt_tokens, (int(0.6*prompt_tokens),))
    prompt_copy[i, idx] = 0
    candidate_40_prompts.append(prompt_copy[i])
candidate_40_prompts = torch.vstack(candidate_40_prompts)
candidate_20_prompts = []
prompt_copy = copy.deepcopy(prompts)
for i in range(T):
    idx = torch.randint(0, prompt_tokens, (int(0.8*prompt_tokens),))
    prompt_copy[i, idx] = 0
    candidate_20_prompts.append(prompt_copy[i])
candidate_20_prompts = torch.vstack(candidate_20_prompts)

pbar = tqdm(total=n_batches)
for batch in range(n_batches):
    idx = torch.arange(batch * args.batch_size,
                       min(T, (batch + 1) * args.batch_size))

    _, _, watermarked_empty_prob, _, watermarked_empty_ntp = generate_watermark(
        prompts[idx], seeds[idx], candidate_98_prompts[idx],
        fixed_inputs=attacked_samples[idx])
    re_calculated_98_probs.append(watermarked_empty_prob)
    if args.method == "transform":
        re_calculated_98_ntps.append(watermarked_empty_ntp.cpu())

    pbar.update(1)
pbar.close()
re_calculated_98_probs = torch.vstack(re_calculated_98_probs)
for itm in range(T):
    re_calculated_98_probs_writer.writerow(
        np.asarray(re_calculated_98_probs[itm].numpy()[:args.tokens_count]))
re_calculated_98_probs_save.close()
np.save(args.save + '-re-calculated-98-ntps.npy', re_calculated_98_ntps)

pbar = tqdm(total=n_batches)
for batch in range(n_batches):
    idx = torch.arange(batch * args.batch_size,
                       min(T, (batch + 1) * args.batch_size))

    _, _, watermarked_empty_prob, _, watermarked_empty_ntp = generate_watermark(
        prompts[idx], seeds[idx], candidate_96_prompts[idx],
        fixed_inputs=attacked_samples[idx])
    re_calculated_96_probs.append(watermarked_empty_prob)
    if args.method == "transform":
        re_calculated_96_ntps.append(watermarked_empty_ntp.cpu())

    pbar.update(1)
pbar.close()
re_calculated_96_probs = torch.vstack(re_calculated_96_probs)
for itm in range(T):
    re_calculated_96_probs_writer.writerow(
        np.asarray(re_calculated_96_probs[itm].numpy()[:args.tokens_count]))
re_calculated_96_probs_save.close()
np.save(args.save + '-re-calculated-96-ntps.npy', re_calculated_96_ntps)

pbar = tqdm(total=n_batches)
for batch in range(n_batches):
    idx = torch.arange(batch * args.batch_size,
                       min(T, (batch + 1) * args.batch_size))

    _, _, watermarked_empty_prob, _, watermarked_empty_ntp = generate_watermark(
        prompts[idx], seeds[idx], candidate_90_prompts[idx],
        fixed_inputs=attacked_samples[idx])
    re_calculated_90_probs.append(watermarked_empty_prob)
    if args.method == "transform":
        re_calculated_90_ntps.append(watermarked_empty_ntp.cpu())

    pbar.update(1)
pbar.close()
re_calculated_90_probs = torch.vstack(re_calculated_90_probs)
for itm in range(T):
    re_calculated_90_probs_writer.writerow(
        np.asarray(re_calculated_90_probs[itm].numpy()[:args.tokens_count]))
re_calculated_90_probs_save.close()
np.save(args.save + '-re-calculated-90-ntps.npy', re_calculated_90_ntps)

pbar = tqdm(total=n_batches)
for batch in range(n_batches):
    idx = torch.arange(batch * args.batch_size,
                       min(T, (batch + 1) * args.batch_size))

    _, _, watermarked_empty_prob, _, watermarked_empty_ntp = generate_watermark(
        prompts[idx], seeds[idx], candidate_80_prompts[idx],
        fixed_inputs=attacked_samples[idx])
    re_calculated_80_probs.append(watermarked_empty_prob)
    if args.method == "transform":
        re_calculated_80_ntps.append(watermarked_empty_ntp.cpu())

    pbar.update(1)
pbar.close()
re_calculated_80_probs = torch.vstack(re_calculated_80_probs)
for itm in range(T):
    re_calculated_80_probs_writer.writerow(
        np.asarray(re_calculated_80_probs[itm].numpy()[:args.tokens_count]))
re_calculated_80_probs_save.close()
np.save(args.save + '-re-calculated-80-ntps.npy', re_calculated_80_ntps)

pbar = tqdm(total=n_batches)
for batch in range(n_batches):
    idx = torch.arange(batch * args.batch_size,
                       min(T, (batch + 1) * args.batch_size))

    _, _, watermarked_empty_prob, _, watermarked_empty_ntp = generate_watermark(
        prompts[idx], seeds[idx], candidate_60_prompts[idx],
        fixed_inputs=attacked_samples[idx])
    re_calculated_60_probs.append(watermarked_empty_prob)
    if args.method == "transform":
        re_calculated_60_ntps.append(watermarked_empty_ntp.cpu())

    pbar.update(1)
pbar.close()
re_calculated_60_probs = torch.vstack(re_calculated_60_probs)
for itm in range(T):
    re_calculated_60_probs_writer.writerow(
        np.asarray(re_calculated_60_probs[itm].numpy()[:args.tokens_count]))
re_calculated_60_probs_save.close()
np.save(args.save + '-re-calculated-60-ntps.npy', re_calculated_60_ntps)

pbar = tqdm(total=n_batches)
for batch in range(n_batches):
    idx = torch.arange(batch * args.batch_size,
                       min(T, (batch + 1) * args.batch_size))

    _, _, watermarked_empty_prob, _, watermarked_empty_ntp = generate_watermark(
        prompts[idx], seeds[idx], candidate_40_prompts[idx],
        fixed_inputs=attacked_samples[idx])
    re_calculated_40_probs.append(watermarked_empty_prob)
    if args.method == "transform":
        re_calculated_40_ntps.append(watermarked_empty_ntp.cpu())

    pbar.update(1)
pbar.close()
re_calculated_40_probs = torch.vstack(re_calculated_40_probs)
for itm in range(T):
    re_calculated_40_probs_writer.writerow(
        np.asarray(re_calculated_40_probs[itm].numpy()[:args.tokens_count]))
re_calculated_40_probs_save.close()
np.save(args.save + '-re-calculated-40-ntps.npy', re_calculated_40_ntps)

pbar = tqdm(total=n_batches)
for batch in range(n_batches):
    idx = torch.arange(batch * args.batch_size,
                       min(T, (batch + 1) * args.batch_size))

    _, _, watermarked_empty_prob, _, watermarked_empty_ntp = generate_watermark(
        prompts[idx], seeds[idx], candidate_20_prompts[idx],
        fixed_inputs=attacked_samples[idx])
    re_calculated_20_probs.append(watermarked_empty_prob)
    if args.method == "transform":
        re_calculated_20_ntps.append(watermarked_empty_ntp.cpu())

    pbar.update(1)
pbar.close()
re_calculated_20_probs = torch.vstack(re_calculated_20_probs)
for itm in range(T):
    re_calculated_20_probs_writer.writerow(
        np.asarray(re_calculated_20_probs[itm].numpy()[:args.tokens_count]))
re_calculated_20_probs_save.close()
np.save(args.save + '-re-calculated-20-ntps.npy', re_calculated_20_ntps)

log_file.close()
