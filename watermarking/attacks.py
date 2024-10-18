import torch

def substitution_attack(tokens, p, vocab_size, distribution=None):
    if distribution is None:
        def distribution(x): return torch.ones(
            size=(len(tokens), vocab_size)) / vocab_size
    idx = torch.randperm(len(tokens))[:int(p*len(tokens))]

    new_probs = distribution(tokens)
    samples = torch.multinomial(new_probs, 1).flatten()
    tokens[idx] = samples[idx]

    return tokens, idx


def deletion_attack(tokens, p):
    idx = torch.randperm(len(tokens))[:int(p*len(tokens))]

    keep = torch.ones(len(tokens), dtype=torch.bool)
    keep[idx] = False
    tokens = tokens[keep]

    return tokens, idx


def is_sublist(sub, main):
    """
    Checks if 'sub' list is a sublist of 'main' list.

    Args:
        sub (list): Sublist to search for.
        main (list): Main list to search within.

    Returns:
        (bool, int): Tuple indicating whether 'sub' is found and the starting index.
    """
    sub_len = len(sub)
    for i in range(len(main) - sub_len + 1):
        if main[i:i + sub_len] == sub:
            return True, i
    return False, -1

def deletion_attack_semantic(tokens, tokenizer):
    """
    Performs a semantic deletion attack by removing the second sentence from the text.

    Args:
        tokens (torch.Tensor): Original token sequence.
        tokenizer: Tokenizer corresponding to the LLM.

    Returns:
        attacked_tokens (torch.Tensor): Token sequence after deletion.
        attack_span (tuple): (attack_start, attack_end) indices in the original tokens.
                              Returns (None, None) if no attack was performed.
    """
    # Decode the original tokens to text
    text = tokenizer.decode(tokens, skip_special_tokens=True)

    # Split the text into sentences
    parts = text.split(". ")

    # If there are less than 2 sentences, cannot perform the defined deletion attack
    if len(parts) < 2:
        return tokens, (None, None)

    # Remove the second sentence (index 1)
    del parts[1]

    # Reconstruct the attacked text
    attacked_text = ". ".join(parts)

    # Encode the attacked text back into tokens
    attacked_tokens = tokenizer.encode(
        attacked_text,
        return_tensors='pt',
        truncation=True,
        max_length=2048
    )[0]

    # This is very very ad-hoc, but it seems that the tokenizer sometimes adds a special token at the beginning
    # of the sequence. If the first token is 1 or 128000, remove it.
    # 128000 for meta-llama/Meta-Llama-3-8B and 1 for mistralai/Mistral-7B-v0.1
    if attacked_tokens[0] == 1 or attacked_tokens[0] == 128000:
        attacked_tokens = attacked_tokens[1:]

    # Convert both token sequences to lists for easier manipulation
    original_tokens = tokens.tolist()
    attacked_tokens_list = attacked_tokens.tolist()

    if len(original_tokens) == len(attacked_tokens_list):
        return tokens, (None, None)

    # Initialize variables to track the start of the attack
    attack_start = 0
    attack_start_addition = 0
    while attack_start_addition < len(attacked_tokens_list) and attacked_tokens_list[attack_start + attack_start_addition] != original_tokens[attack_start]:
        attack_start_addition += 1
    attack_end = None

    # Find the first token that differs between the original and attacked tokens
    for i in range(len(attacked_tokens_list) - attack_start_addition):
        if i >= len(original_tokens) or attacked_tokens_list[i + attack_start_addition] != original_tokens[i]:
            attack_start = i
            break
    else:
        return attacked_tokens, (len(attacked_tokens_list), len(original_tokens))

    found = False

    for i in range(attack_start + 1 + attack_start_addition, len(attacked_tokens_list)):
        # Extract the remaining attacked tokens after the attack_start
        remaining_attacked = attacked_tokens_list[i:]

        # Check if the remaining_attacked tokens are a sublist of the original tokens
        found, index = is_sublist(remaining_attacked, original_tokens)

        if found and index > attack_start:
            attack_end = index
            break
        elif found:
            found = False

    if not found:
        # If not found, set attack_end to the end of the original tokens
        attack_end = len(original_tokens)

    # Return the span of the attack in the original tokens
    attack_span = (attack_start, attack_end)

    return attacked_tokens, attack_span


def insertion_attack(tokens, p, vocab_size, distribution=None):
    if distribution is None:
        def distribution(x): return torch.ones(
            size=(len(tokens), vocab_size)) / vocab_size
    idx = torch.randperm(len(tokens))[:int(p*len(tokens))]

    new_probs = distribution(tokens)
    samples = torch.multinomial(new_probs, 1)
    for i in idx.sort(descending=True).values:
        tokens = torch.cat([tokens[:i], samples[i], tokens[i:]])
        tokens[i] = samples[i]

    return tokens, idx


def insertion_attack_semantic(tokens, prompt, tokenizer, model, max_insert_length=50):
    """
    Performs a semantic insertion attack by inserting a generated sentence between the first and second sentences.

    Args:
        tokens (torch.Tensor): Original token sequence.
        prompt (torch.Tensor): Prompt token sequence to prepend before generation.
        tokenizer (transformers.PreTrainedTokenizer): Tokenizer corresponding to the LLM.
        model: Pre-loaded language model for text generation.
        max_insert_length (int, optional): Maximum number of tokens to generate for the insertion.
                                            Defaults to 50.

    Returns:
        attacked_tokens (torch.Tensor): Token sequence after insertion.
        attack_span (tuple): (attack_start, attack_end) indices in the original tokens.
                              Returns (None, None) if no attack was performed.
    """
    device = next(model.parameters()).device  # Get device from the model
    # Decode the original tokens to text
    text = tokenizer.decode(tokens, skip_special_tokens=True)

    # Split the text into sentences using a more robust method if needed
    parts = text.split(". ")

    # If there are fewer than 2 sentences, cannot perform the defined insertion attack
    if len(parts) < 2:
        return tokens, (None, None)

    # Extract the first sentence to use as a prefix for generation
    prefix = parts[0].strip() + ". "

    # Generate a new sentence to insert
    with torch.no_grad():
        # Encode the prefix
        prefix_ids = tokenizer.encode(
            prefix, return_tensors='pt', truncation=True, max_length=2048
        ).to(device)

        # Ensure prompt is 2D
        if prompt.dim() == 1:
            prompt = prompt.unsqueeze(0)
        elif prompt.dim() != 2:
            raise ValueError(f"Prompt tensor must be 1D or 2D, but got {prompt.dim()}D.")

        # Move prompt to the same device as model and input_ids
        prompt = prompt.to(device)

        # Concatenate prompt and prefix_ids along the sequence dimension (dim=1)
        input_ids = torch.cat([prompt, prefix_ids], dim=1)

        # Generate tokens until a period is generated to ensure a complete sentence
        generated_ids = model.generate(
            input_ids,
            max_length=input_ids.shape[1] + max_insert_length,  # Adjust max_length as needed
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id
        )

        # Extract the generated text after the prefix
        generated_text = tokenizer.decode(
            generated_ids[0][input_ids.shape[1]:], skip_special_tokens=True
        )

        # Optionally, truncate the generated text at the first period to ensure a complete sentence
        period_index = generated_text.find('.')
        if period_index != -1:
            new_sentence = generated_text[:period_index + 1].strip()
        else:
            new_sentence = generated_text.strip() + "."

    # Insert the new sentence between the first and second sentences
    inserted_text = new_sentence + " "
    # Reconstruct the attacked text
    attacked_text = parts[0].strip() + ". " + inserted_text + ". ".join(parts[1:])

    # Encode the attacked text back into tokens
    attacked_tokens = tokenizer.encode(
        attacked_text,
        return_tensors='pt',
        truncation=True,
        max_length=2048
    )[0].to(device)

    attack_start = len(tokenizer.encode(parts[0], return_tensors='pt', truncation=True, max_length=2048)[0])
    attack_end = len(tokenizer.encode(parts[0].strip() + ". " + inserted_text, return_tensors='pt', truncation=True, max_length=2048)[0])

    # Define the attack span as the insertion point in the original tokens
    attack_span = (attack_start, attack_end)

    return attacked_tokens, attack_span
