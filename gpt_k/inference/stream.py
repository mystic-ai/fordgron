from typing import List
import torch
import copy
import torch.nn.functional as F

def stream(
    model,
    args,
    context_tokens: List[List[int]],
    eos_token_id: int = None,
    maximum_tokens: int = None,
    recompute: bool = False,
    temperature: float = 0.0,
    top_k: int = 0,
    top_p: float = 0.0
):
    """
    iterator producing text completions
    args: NeoXArgs.
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

    model.eval()

    # convert to tensor and broadcast
    context_tokens = torch.LongTensor(context_tokens).to(1)

    # get attention mask / position ids
    context_tokens, attention_mask, position_ids = get_batch(args, context_tokens)

    # set variables
    eos_token_id = 0
    maximum_tokens = maximum_tokens or (
        2048 - token_generation_start_index.max().item() - 1
    )
    batch_size = context_tokens.size(0)

    with torch.no_grad():
        # initialize generation variables
        state_is_done = torch.zeros([batch_size]).byte().to(1)
        layer_past = None
        layer_past_length = 0

        for _ in range(context_tokens.size(1), 2000):
            if recompute:  # recompute all tokens
                logits = model(context_tokens, layer_past=layer_past)
                if logits is not None:  # if pipe parallel, not all ranks return logits
                    generated_token_logits = logits[
                        :, token_index_to_generate - 1, :
                    ]  # [bs, seq, vocab_size] -> [bs, vocab_size]
            else:  # use kv cache         
                logits, layer_past = model(context_tokens, layer_past=layer_past)
                if logits is not None:  # if pipe parallel, not all ranks return logits
                    generated_token_logits = (
                        logits[:, -1].view(batch_size, -1).contiguous()
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

                context_tokens = generated_tokens.unsqueeze(0)
            yield generated_tokens

def get_batch(neox_args, context_tokens: torch.Tensor):
    """
    Generate batch from context tokens. Attention mask and position ids are created. Returned tensors will be on CUDA.
    neox_args: NeoXArgs.
    context_tokens: torch tensor with dimensions [batch, context_size]
    returns: tuple of torch tensors (tokens, attention_mask, position_ids) on CUDA
    """

    # Move to GPU.
    tokens = context_tokens.contiguous().to(1)
    # Get the attention mask and position ids.
    attention_mask, _, position_ids = get_ltor_masks_and_position_ids(
        data=tokens,
        eod_token=0 #eod token
    )
    return tokens, attention_mask, position_ids


def find_longest_list(list):
  list_lengths = [len(i) for i in list]
  return max(list_lengths)


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


def switch(val1, val2, boolean):
    """
    replaces items in val1 with items in val2 where boolean = True
    """
    boolean = boolean.type_as(val1)
    return (1 - boolean) * val1 + boolean * val2

def get_attn_mask(seq_length, device):
    """
    Get triangular attention mask for a given sequence length / device.
    """
    # lower triangular attention mask
    mask = torch.tril(torch.ones((1, seq_length, seq_length), device=device)).view(
        1, 1, seq_length, seq_length
    )

    # convert to binary
    return mask < 0.5


def get_ltor_masks_and_position_ids(
    data,
    eod_token,
):
    """Build masks and position id for left to right model."""

    # Extract batch size and sequence length.
    batch_size, seq_length = data.size()

    # Attention mask (lower triangular).
    attention_mask = get_attn_mask(
        seq_length=seq_length,
        device=data.device,
    )

    # Loss mask.
    loss_mask = torch.ones(data.size(), dtype=torch.float, device=data.device)

    # Position ids.
    position_ids = torch.arange(seq_length, dtype=torch.long, device=data.device)
    position_ids = position_ids.unsqueeze(0).expand_as(data)

    return attention_mask, loss_mask, position_ids