import torch


def generate(model, prompts, vocab_size, n, m, seeds, key_func, sampler, random_offset=True, empty_prompts=None):
    batch_size = len(prompts)

    generator = torch.Generator()
    xis, pis = [], []
    for seed in seeds:
        generator.manual_seed(int(seed))
        xi, pi = key_func(generator, n, vocab_size)
        xis.append(xi.unsqueeze(0))
        pis.append(pi.unsqueeze(0))
    xis = torch.vstack(xis)
    pis = torch.vstack(pis)

    # deliberately not controlling this randomness with the generator
    if random_offset:
        offset = torch.randint(n, size=(batch_size,))
    else:
        offset = torch.zeros(size=(batch_size,), dtype=torch.int64)

    inputs = prompts.to(model.device)
    empty_inputs = empty_prompts.to(model.device)

    sampling_probs = torch.zeros((batch_size, 0)).to(model.device)
    empty_sampling_probs = torch.zeros((batch_size, 0)).to(model.device)

    attn = torch.ones_like(inputs)

    past = None
    empty_past = None

    for i in range(m):
        with torch.no_grad():
            if past:
                output = model(
                    inputs[:, -1:], past_key_values=past, attention_mask=attn)
                empty_output = model(
                    empty_inputs[:, -1:], past_key_values=empty_past, attention_mask=attn)
            else:
                output = model(inputs)
                empty_output = model(empty_inputs)

        probs = torch.nn.functional.softmax(output.logits[:, -1], dim=-1).cpu()
        empty_probs = torch.nn.functional.softmax(
            empty_output.logits[:, -1], dim=-1).cpu()

        tokens, sampling_prob = sampler(probs, pis, xis[torch.arange(
            batch_size), (offset.squeeze()+i) % n])
        tokens = tokens.to(model.device)
        empty_sampling_prob = torch.gather(empty_probs, 1, tokens)
        sampling_prob = sampling_prob.to(model.device)

        sampling_probs = torch.cat([sampling_probs, sampling_prob], dim=-1)
        empty_sampling_probs = torch.cat(
            [empty_sampling_probs, empty_sampling_prob], dim=-1)

        inputs = torch.cat([inputs, tokens], dim=-1)
        empty_inputs = torch.cat([empty_inputs, tokens], dim=-1)

        past = output.past_key_values
        empty_past = empty_output.past_key_values

        attn = torch.cat([attn, attn.new_ones((attn.shape[0], 1))], dim=-1)

    return inputs.detach().cpu(), sampling_probs.detach().cpu(), empty_sampling_probs.detach().cpu()

# generate unwatermarked completions of token length m given list of prompts


def generate_rnd(prompts, m, model, empty_prompts):
    inputs = prompts.to(model.device)
    empty_inputs = empty_prompts.to(model.device)

    attn = torch.ones_like(inputs)

    past = None
    empty_past = None

    sampling_probs = torch.zeros((inputs.shape[0], 0)).to(model.device)
    empty_sampling_probs = torch.zeros((inputs.shape[0], 0)).to(model.device)

    for i in range(m):
        with torch.no_grad():
            if past:
                output = model(
                    inputs[:, -1:], past_key_values=past, attention_mask=attn)
                empty_output = model(
                    empty_inputs[:, -1:], past_key_values=empty_past, attention_mask=attn)
            else:
                output = model(inputs)
                empty_output = model(empty_inputs)

        probs = torch.nn.functional.softmax(output.logits[:, -1], dim=-1)
        empty_probs = torch.nn.functional.softmax(
            empty_output.logits[:, -1], dim=-1)

        tokens = torch.multinomial(probs, 1)

        sampling_prob = torch.gather(probs, 1, tokens)
        empty_sampling_prob = torch.gather(empty_probs, 1, tokens)

        sampling_probs = torch.cat([sampling_probs, sampling_prob], dim=-1)
        empty_sampling_probs = torch.cat(
            [empty_sampling_probs, empty_sampling_prob], dim=-1)

        inputs = torch.cat([inputs, tokens], dim=1)
        empty_inputs = torch.cat([empty_inputs, tokens], dim=1)

        past = output.past_key_values
        empty_past = empty_output.past_key_values

        attn = torch.cat([attn, attn.new_ones((attn.shape[0], 1))], dim=-1)

    return inputs.detach().cpu(), sampling_probs.detach().cpu(), empty_sampling_probs.detach().cpu()
