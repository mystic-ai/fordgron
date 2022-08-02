from typing import List
import torch
import copy
import torch.nn.functional as F
from torch import nn
import time
from rich.progress import Progress


def top_k_logits(logits, k):
    if k == 0:
        return logits
    values, _ = torch.topk(logits, k)
    min_values = values[:, -1]
    return torch.where(
        logits < min_values, torch.ones_like(logits, dtype=logits.dtype) * -1e10, logits
    )


def stream_token_ids(model, inference_kwargs, input_tokens=None, model_seq_len=2048):
    context = torch.tensor(
        input_tokens, device=inference_kwargs["device"], dtype=torch.long
    )
    prev = context
    output = context
    previous_generation = None
    with torch.no_grad():
        for i in range(inference_kwargs["generation_length"]):
            # logits, past = model(prev, past_key_values=past)
            logits, previous_generation = model(
                prev,
                past_key_values=previous_generation,
                use_cache=inference_kwargs["use_cache"],
            )
            res = logits[:, -1].argmax(-1)
            if inference_kwargs["use_cache"]:
                prev = res.unsqueeze(0)
            else:
                prev = torch.cat((prev, res.unsqueeze(0)), dim=-1)
            yield res
