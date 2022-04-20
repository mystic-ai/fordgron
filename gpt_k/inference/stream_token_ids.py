from typing import List
import torch
import copy
import torch.nn.functional as F
from torch import nn
import time
from rich.progress import Progress

def stream_token_ids(
    model: nn.Module,
    inference_kwargs: dict,
    prompt_token_ids: List[List[int]],
    model_seq_len: int
):
    """
    Args:
        model: the model to be used for inference. it should already be loaded onto GPU, with the weights loaded in.
        inference_kwargs: arguments to be used in this specific inference run. these typically relate to how generation should auto-end or how logits are sampled.
        prompt_token_ids: the first input to the transformer, of size [batch_len, seq_len]. this stream handles all generations after the first in a loop.
        model_seq_len: the width of the model's context window (typically 2048 tokens). after training this cannot be changed, although finetuning might be able to update it.
    
    Returns:
        yields the token(s) generated by the model on each forward pass. until the requested number of tokens has been fulfilled, the model will append the generated token to the prompt for the next autoregressive forward pass.
    """

    # place the model into evaluation mode
    model.eval()

    # convert to tensor
    prompt_token_ids = torch.IntTensor(prompt_token_ids).to(1)

    batch_len, input_seq_len = prompt_token_ids.size()

    with torch.no_grad():
        # initialise layer_past as an empty variable, for it will be populated on future forward passes
        layer_past = None

        num_tokens_to_generate = inference_kwargs["generation_length"] - input_seq_len
        with Progress() as progress:
            task1 = progress.add_task("inferencing", total=num_tokens_to_generate)
            for _ in range(num_tokens_to_generate):
                if inference_kwargs["use_cache"]:
                    logits, layer_past = model(prompt_token_ids, layer_past=layer_past) # logits: [batch_len, seq_len, vocab_len] # 0.029_550_552368164062 s
                    # everything after this point is taking 0.04563331604003906 s.
                    # collapse the sequence length dimension, because it will be 1 anyway
                    generated_token_logits = (
                        logits[:, -1].view(batch_len, -1).contiguous()
                    ) # [batch_len, vocab_len] # 0.000_015_735626220703125 s

                # sample token id of the to be generated token
                # if sampling is set to be deterministic (aka greedy decoding) then simply return the logit with the highest probability
                if inference_kwargs["temperature"] == 0 and inference_kwargs["top_k"] == 0 and inference_kwargs["top_p"] == 0:
                    generated_token_ids = torch.argmax(
                        generated_token_logits, dim=-1
                    ).view(-1) # [batch_len]
                else:
                    generated_token_logits = generated_token_logits.float() # 0.000_015_020370483398438 s

                    if inference_kwargs["temperature"] > 0:
                        # divide the logits by the temperature, this scales logit confidence downwards (or upwards), taking inspiration from thermodynamics
                        # 0.000_460_62469482421875 s
                        generated_token_logits = generated_token_logits / inference_kwargs["temperature"]

                    generated_token_logits = filter_logits(
                        generated_token_logits, top_k=inference_kwargs["top_k"], top_p=inference_kwargs["top_p"]
                    ) # 0.044_772_62496948242 s

                    # spread the logits so they're in between 0 and 1, so the next stage of sampling is super easy
                    generated_token_ids = F.softmax(generated_token_logits, dim=-1) # 0.000_012_159347534179688 s
                    # categorical distribution
                    generated_token_ids = torch.multinomial(
                        generated_token_ids, num_samples=1
                    ).view(-1) # [batch_size] 0.000_165_2240753173828 s

                # add the generated token id back into the prompt
                prompt_token_ids = generated_token_ids.unsqueeze(0)
                progress.update(task1, advance=1)
                yield generated_token_ids


def filter_logits(logits: torch.Tensor, top_k:int = 0, top_p: float = 0.0, filter_value: float = -float("Inf")):
    """
        Args:
            logits: the raw logits out of the model, typically in shape [batch_len, vocab_len]
            top_k: instructs the filter to remove all but the top 'k' highest probability logits, this acts as a window from which to sample
            top_p: a 'smart' way of rewarding confidence in the model, this is a threshold using cumulative probabilities, so high confidence leads to smaller sampling window, low confidence leads to a larger one
            filter_value: the value to replace masked logits with. typically the default of -infinity is the right choice
        
        Returns:
            filtered logits, having had top_k and top_p filtering both applied. it's important to emphasise that these are not directly sampling methods, but they reduce the sampling window
    """
    # pick the smaller from the supplied top_k value and the vocab_len of the model
    top_k = min(top_k, logits.size(-1))
    if top_k > 0:
        now = time.time()
        top_k_values, _ = torch.topk(logits, top_k) # (values Tensor, indices Tensor) 0.000_415_3251647949219 s
        # Remove all tokens with a probability less than the last token of the top-k

        # indices_to_remove is a tensor with True in all elements where the equality operator returns true
        indices_to_remove = torch.lt(logits, top_k_values[..., -1, None]) # [batch_len, vocab_len] 0.000_017_642974853515625 s
        # since the topk torch function orders values in descending order, we can query for which logits are less than the final element of the values tensor
        # the None does the equivalent of unsqueeze(0)

        # on the original logits tensor, set all 'True' (aka non-top-k) elements to an extreme negative number
        logits[indices_to_remove] = filter_value # 0.000_016_689300537109375 s

    if top_p > 0.0:
        # during this sampling technique we select the smallest set of tokens whose cumulative probability is more than or equal to top_p. this balances certainty and uncertainty in a pleasant way
        descending_logits_values, descending_logits_indices = torch.sort(logits, descending=True, dim=-1) # 0.000_080_34706115722656 s
        normalised_descending_logits_values = F.softmax(descending_logits_values, dim=-1) # 0.000_010_49041748046875 s()
        cumulative_probabilities = torch.cumsum(normalised_descending_logits_values, dim=-1) # [batch_len, vocab_len] every element in the tensor is the sum of all preceding elements and itself 0.000_094_89059448242188 s

        # top_p is a threshold, select the logits below that threshold 
        sorted_indices_to_remove = torch.ge(cumulative_probabilities, top_p) # [batch_len, vocab_len] 0.000_014_30511474609375 s
        
        # at index 0 we insert a 0 so that even with the smallest top_k possible, at least one token will be picked
        # this means we need to 'shift' all the other index probabilities down one, but remember that because the values are sorted in descending order this will simply shift token i's cumprob to token i+1's cumprob
        # 0.000_512_8383636474609 s
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # for each sequence in the batch, replace the masked elements (i.e. not within top_p) with the very huge negative number to minimise the sampling of them
        for batch_index in range(descending_logits_indices.size(0)):
            now = time.time()
            ya = descending_logits_indices[batch_index]
            now = time.time()
            ba = sorted_indices_to_remove[batch_index]
            indices_to_remove = descending_logits_indices[batch_index][sorted_indices_to_remove[batch_index]] # 0.044_241_42837524414 s THIS IS THE SLOW OFFENDING LINE
            logits[batch_index][indices_to_remove] = filter_value # 0.000_038_38539123535156 s

    return logits