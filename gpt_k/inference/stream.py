import torch
import torch.nn.functional as F
import copy
from typing import List

def stream(
    model,
    prompt_token_ids: List[List[int]],
    eos_token_id: int = None,
    eod_token_id: int = None,
    maximum_tokens: int = None,
    recompute: bool = False,
    temperature: float = 1.0,
    top_k: int = 50,
    top_p: float = 1.0,
    stop_tokens=None,
):
    """
    iterator producing text completions
    neox_args: NeoXArgs.
    model: a Megatron model.
    context_tokens: the prompt to complete; unpadded list of lists of tokens ids
    context_lengths: lengths of context tokens of dimension [batch]; the context length records for each bach item how many non-padded tokens are provided
    eos_token_id: end of text token at which completion is terminated, even if max_tokes count has not been reached
    attention_mask: attention mask for megatron model.
    position_ids: position ids for positional encoding.
    maximum_tokens: maximum number of tokens to be generated; careful! if a batch input is provided maximum_tokens specifies the maximum number of forwards.
                    longer batch items get less generated tokens.
    recompute: flag indicating whether a cache is used for already forwarded tokens (true) or whether all tokens are recomputed at every iteration (false)
    temperature (default 0.0): exponential scaling output distribution ("higher == more risk")
    top_k (default 0): integer -> integer between 0 and the models vocab size. Filters out any logits with a probability less than that of the top_kth token.
    top_p (default 0.0): float -> Top-p (nucleus) sampling chooses from the smallest possible set of tokens whose cumulative probability exceeds the probability top_p.
    note: greedy decoding is used if temperature is 0.0, top_k is 0 and top_p is 0.0
    yields: (
                tokens (completions from model),
                token_generation_start_index (token index per batch item for the first generated token),
                token_generation_end_index (token index per batch item for the last generated token),
                logits (logits which are so far computed, zeros otherwise),
                is_done (flag for each bach item indicating whether an eod token was generated)
            )
            * each iteration adds a generated token to the context_tokens
            * output contains both context_tokens from input and generated tokens
            * if batch items have different lengths, the iterator will start at the first completion and return the unchanged input context token otherwise
    """

    seq_len = 2048

    model.eval()

    # pad batch in order to allow conversion to tensor
    prompt_token_ids, prompt_lengths = pad_batch(
        copy.deepcopy(prompt_token_ids),
        pad_id=eod_token_id,
        pad_len=seq_len,
    )
    # prompt_token_ids: List[int] [batch_len, pad_len] e.g. [2048, 2048, 2048]
    # prompt_lengths: List[int] [batch_len, input_seq_lens] e.g. [12, 180, 56]

    # convert to tensor
    prompt_token_ids = torch.cuda.LongTensor(prompt_token_ids, device=1)
    if stop_tokens:
        stop_tokens = torch.cuda.LongTensor(stop_tokens)
        if stop_tokens.ndim == 1:
            stop_tokens = stop_tokens.unsqueeze(0)

    # the prompt_lengths array is also an array of starting indices for generation (if input_seq_len = 12, starting index is 12 (it's position 13)
    token_generation_start_index = torch.cuda.LongTensor(prompt_lengths, device=1)

    # get attention mask / position ids
    prompt_token_ids, attention_mask, position_ids = get_batch(prompt_token_ids, eod_token_id)
    # prompt_token_ids: torch.Tensor [batch_len, seq_len]
    # attention_mask: torch.Tensor [1, 1, seq_len, seq_len]
    # position_ids: torch.Tensor [batch_len, seq_len] simply ascending numbers

    # set variables
    # for maximum_tokens, either use the strict enforced maximum_tokens passed into stream function
    # or find the difference between seq_len and the largest prompt length (aka the smallest distance)
    # so the largest prompt sets 'the maximum amount of generated tokens' as all other generations will follow it up to its completion index
    maximum_tokens = maximum_tokens or (
        seq_len - token_generation_start_index.max().item() - 1
    )
    batch_len = prompt_token_ids.size(0)

    # what's the smallest prompt length in the batch? because all generations will start from that prompt_length + 1 position, even if it has a longer prompt length.
    token_index_to_generate = token_generation_start_index.min().item()
    first_token_index_to_generate = token_index_to_generate

    # what's smaller: the model's seq_len, or (the smallest prompt length + the maximum number of tokens to generate -1)? use that as the final token index to generate
    last_token_index_to_generate = min(
        seq_len
        - 1,  # never generate more than the model's sequence length
        token_index_to_generate + maximum_tokens - 1,
    )

    with torch.no_grad():
        # initialize generation variables
        state_is_done = torch.zeros([batch_len]).byte().cuda(1)
        token_generation_end_index = torch.ones([batch_len]).long().cuda(1) * (-1)

        while token_index_to_generate <= last_token_index_to_generate:
            if token_index_to_generate == first_token_index_to_generate:
                tokens_to_use = prompt_token_ids[:, :token_index_to_generate]
                positions_to_use = position_ids[:, :token_index_to_generate]
            else:
                tokens_to_use = prompt_token_ids[:, token_index_to_generate - 1].view(
                    batch_len, -1
                )
                positions_to_use = position_ids[
                    :, token_index_to_generate - 1
                ].view(batch_len, -1)

            model_inputs = (
                tokens_to_use,  # input_ids
                positions_to_use,  # position_ids
                attention_mask,  # attention_mask
            )
            # update this
            logits = model(tokens_to_use, attention_mask)
            if logits is not None:  # if pipe parallel, not all ranks return logits
                generated_token_logits = (
                    logits[:, -1].view(batch_len, -1).contiguous()
                )  # [bs, seq, vocab_size] -> [bs, vocab_size]

            if logits is not None:
                # sample token id of the to be generated token
                if temperature == 0.0 and top_k == 0 and top_p == 0.0:
                    generated_tokens = torch.argmax(
                        generated_token_logits, dim=-1
                    ).view(-1)
                else:
                    generated_token_logits = generated_token_logits.float()
                    if temperature > 0.0:
                        generated_token_logits /= temperature
                    generated_token_logits = filter_logits(
                        generated_token_logits, top_k=top_k, top_p=top_p
                    )
                    next_token_log_probs = F.softmax(generated_token_logits, dim=-1)
                    generated_tokens = torch.multinomial(
                        next_token_log_probs, num_samples=1
                    ).view(-1)

            # determine if state has started for each batch item
            state_started = (
                token_generation_start_index <= token_index_to_generate
            )  # check which batch items have been started

            # switch out padding tokens for generated tokens
            prompt_token_ids[:, token_index_to_generate] = switch(
                prompt_token_ids[:, token_index_to_generate].view(-1),
                generated_tokens,
                state_started,
            )

            # determine if state has finished for each batch item
            state_done = (
                generated_tokens == eos_token_id
            ).byte() & state_started.byte()  # check which batch items produce an eos_token in the current iteration
            state_just_finished = (state_done & ~state_is_done).bool()
            state_is_done = state_is_done | state_done
            stop_tokens_produced = torch.zeros_like(state_is_done)
            for batch_idx, ctx in enumerate(prompt_token_ids):
                stop_tokens_produced[batch_idx] = stop_tokens_in_completion(
                    stop_tokens, prompt_token_ids, batch_idx, token_index_to_generate
                )
            state_is_done = state_is_done | stop_tokens_produced

            token_generation_end_index[
                (state_started.byte() & ~state_is_done).bool()
            ] = token_index_to_generate

            token_index_to_generate += 1

            yield prompt_token_ids, token_generation_start_index, token_generation_end_index, state_is_done.bool()
            if torch.all(state_is_done):
                break

def get_batch(prompt_token_ids, eod_token_id):
    """
    Generate batch from prompt tokens. Attention mask and position ids are created. Returned tensors will be on CUDA.
    neox_args: NeoXArgs.
    prompt_tokens: torch tensor with dimensions [batch, prompt_size]
    returns: tuple of torch tensors (tokens, attention_mask, position_ids) on CUDA
    """

    # Move to GPU.
    tokens = prompt_token_ids.contiguous().cuda(1)
    # Get the attention mask and position ids.
    attention_mask, _, position_ids = get_ltor_masks_and_position_ids(
        data=tokens,
        eod_token=eod_token_id,
        eod_mask_loss=False,
    )
    return tokens, attention_mask, position_ids


def pad_batch(prompt_token_ids: List[List[int]], pad_id: int, pad_len: int):
    """
    pads prompt lengths in prompt_token_ids with pad_id to equal neox_args.seq_length,
    and returns the padded batch and the new lengths.
    prompt_token_ids: list of lists of tokens
    pad_id: int, integer to use as padding token
    pad_len: int, prompt length to be padded; all batch items will be padded to the same length
    returns: tuple of padded prompt tokens and a list of unpadded token count
    """

    prompt_lengths = []
    for tokens in prompt_token_ids:
        prompt_length = len(tokens)
        if prompt_length < pad_len:
            tokens.extend([pad_id] * (pad_len - prompt_length))
        elif prompt_length > pad_len:
            raise ValueError("prompt_length is bigger than to be padded length")
        prompt_lengths.append(prompt_length)
    return prompt_token_ids, prompt_lengths

def get_ltor_masks_and_position_ids(
    data,
    eod_token,
    eod_mask_loss=False,
):
    """Build masks and position id for left to right model."""

    # Extract batch size and sequence length._
    _, seq_len = data.size()
    print(seq_len)

    # Attention mask (lower triangular).
    attention_mask = get_attn_mask(
        seq_len,
        device=data.device
    ) # [1, 1, seq_len, seq_len]

    print("the mask has size")
    print(attention_mask.size())

    # Loss mask.
    loss_mask = torch.ones(data.size(), dtype=torch.float, device=data.device)
    if eod_mask_loss:
        loss_mask[data == eod_token] = 0.0

    # Position ids.
    position_ids = torch.arange(seq_len, dtype=torch.long, device=data.device)
    position_ids = position_ids.unsqueeze(0).expand_as(data)

    return attention_mask, loss_mask, position_ids

def get_attn_mask(seq_len, device):
    return torch.tril(torch.ones((1, seq_len, seq_len), device=device)).view(1, 1, seq_len, seq_len) < 0.5

def switch(val1, val2, boolean):
    """
    replaces items in val1 with items in val2 where boolean = True
    """
    boolean = boolean.type_as(val1)
    return (1 - boolean) * val1 + boolean * val2

def stop_tokens_in_completion(stop_tokens, prompt_token_ids, batch_index, current_index):
    if stop_tokens is None:
        return False
    res = []
    for token_group in stop_tokens:
        prompt = prompt_token_ids[batch_index, : current_index + 1]
        prompt = prompt[-len(token_group) :]
        if prompt.shape[0] == token_group.shape[0]:
            res.append(all(token_group == prompt))
        else:
            res.append(False)
    return any(res)

def filter_logits(logits, top_k=0, top_p=0.0, filter_value=-float("Inf")):
    """
    Filters the logits using top_k / top_p, filling any filtered vocab items with filter_value (defaults to -inf).
    This function has been mostly taken from huggingface conversational ai code at
    https://medium.com/huggingface/how-to-build-a-state-of-the-art-conversational-ai-with-transfer-learning-2d818ac26313
    logits: torch.Tensor -> logits of megatron model.
    top_k: integer -> integer between 0 and the models vocab size. Filters out any logits with a probability less than that of the top_kth token.
    top_p: float -> Top-p (nucleus) sampling chooses from the smallest possible set of tokens whose cumulative probability exceeds the probability top_p.
    returns: (filtered) logits"""

    if top_k > 0:
        # Remove all tokens with a probability less than the
        # last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        # convert to 1D
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token
        # above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        for i in range(sorted_indices.size(0)):
            indices_to_remove = sorted_indices[i][sorted_indices_to_remove[i]]
            logits[i][indices_to_remove] = filter_value

    return logits