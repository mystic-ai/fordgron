import torch.nn as nn
import torch
import math
from rich.progress import Progress

from .modules import TransformerBlock


class Transformer(nn.Module):
    def __init__(self, args, use_cache=False, device=None):
        super().__init__()

        """
        if use_cache is True, then the keys and values from the previous transformer block are saved and sent to the next transformer block
        by preventing this extra computation, inference can be a little faster at the cost of model performance (I believe, although there seems to be 0 research on this)
        """
        self.use_cache = use_cache

        with Progress() as progress:
            task1 = progress.add_task("building transformer", total=args.num_layers + 3) # depth + three other steps

            """
            embedding layer which acts as a fancy lookup table, projecting the token id (integer) to a vector with dimension embedding_dim
            since this embedding vector is trained with the model, it has learned 'representations' of that token by how it relates to others, aka 'the meaning of' that token
            """
            self.token_id_embedding = nn.Embedding(args.vocab_size, args.hidden_size, device=device)
            progress.update(task1, advance=1)

            """
            the core of the transformer is a stack of transformer blocks, each of which essentially applies self-attention (and a few other tricks) to the output of the previous block
            atm this is a decoder-only architecture like GPT-2, but eventually it will be extended to include encoders
            the first block receives the 'embedded' tokens, and each block ouptuts a tensor with that same shape
            """
            self.decoder_stack = nn.ModuleList([])
            for layer_i in range(args.num_layers):
                self.decoder_stack.append(TransformerBlock(args, use_cache, device=device))
                progress.update(task1, advance=1)

            """
            after the final transformer block there is an additional layer norm (pattern set by GPT-2)
            supposedly this creates norm invariance without requiring weight standardisation (in other words, it makes training more stable)
            """
            self.final_layer_norm = nn.LayerNorm(
                args.hidden_size,
                eps=args.layernorm_epsilon,
                device=device,
            )
            progress.update(task1, advance=1)

            """
            finally the model needs to 'reverse the embedding' by projecting the hidden states of attended tokens out into a vector with dimension vocab_len
            at this stage the model outputs logits which we can sample in order to select the next token in the generation process, bearing in mind that the next token must be in the vocab (and therefore its token id will be represented by one item in the vocab_len vector)
            """
            self.logits = nn.Linear(
                args.hidden_size,
                args.vocab_size,
                bias=False,
                device=device,
            )
            progress.update(task1, advance=1)

    def forward(self, X, attention_mask=None, layer_past=None):
        """
        for every new token that requires generation, the forward function will be called on the transformer
        this is because one-token-at-a-time generation is setup, and the state of the transformer needs to be reset (bar caching) on each new token request
        it would be interesting to explore how two-tokens-at-a-time generation would behave

        X: torch.Tensor [batch_len, input_seq_len] int = token ids from the tokenizer
        attention_mask: torch.Tensor [?] bool = shows 'True' at the indices of the 'visible' token id and 'False' at the indices of the 'masked' token id
        layer_past: torch.Tensor [?] ? = attention keys and values from the previous transformer block, so they can be reused 
        """

        """
        atm a new mask is built on each forward pass, although it is likely possible that they can be cached (but this should be benchmarked)
        attention_mask is a tensor of Trues and Falses where the indices of all masked tokens are False
        """
        if attention_mask is None:
            attention_mask = generate_mask(X.size(1)).to(X.device)

        """
        if caching of keys and values from the previous forward pass is enabled
        on the first forward pass there will be no layer_past, but all subsequent generations will be able to use the cache
        """
        if self.use_cache:
            if layer_past is None:
                """
                if there is no cached layer, the kv_length is the length of the longest input sequence (add this)
                """
                kv_length = X.size(1) # input_seq_len NOTE: currently this implementation does not allow for multi-batch inference
            else:
                """
                if there is a cached previous layer, then we can't get kv_length from this forward pass's input as that will only be one token long
                so we must grab the 'true' previous seq_len from the cached layer size and add one to account for the current one
                """
                kv_length = layer_past[0].size(1) + 1 # previous input_seq_len + 1
            attention_mask = attention_mask[..., :X.shape[1], :kv_length]

        """
        if there is no historic generation data, then create a fake
        """
        if layer_past is None:
            layer_past = [None] * len(self.decoder_stack)
        kv_cache_list = []

        """
        setup is complete, now we need to pass the inputs through the transformer. see the init function for details on what each module does
        X = [batch_len, input_seq_len]
        """

        hidden_states = self.token_id_embedding(X) # [batch_len, input_seq_len, embedding_dim]

        hidden_states = self.swap_dimensions(hidden_states, 0, 1) # [input_seq_len, batch_len, embedding_dim]

        """
        we must manually pass the hidden states through each layer of the stack in order to set the true layer_past and extract the kv_cache
        """
        for layer_i, transformer_block in enumerate(self.decoder_stack):
            hidden_states, kv_cache = transformer_block(
                hidden_states,
                attention_mask=attention_mask,
                layer_past=layer_past[layer_i],
            ) # [input_seq_len, batch_len, embedding_dim]
            kv_cache_list.append(kv_cache)

        hidden_states = self.swap_dimensions(hidden_states, 0, 1) # [batch_len, input_seq_len, embedding_dim]

        hidden_states = self.final_layer_norm(hidden_states) # [batch_len, input_seq_len, embedding_dim]

        logits = self.logits(hidden_states) # [batch_len, input_seq_len, vocab_len]

        """
        only return the logits if kv_caching isn't on
        """
        if self.use_cache:
            return logits, kv_cache_list
        else:
            return logits

    @classmethod
    def swap_dimensions(cls, X, dim_1, dim_2):
        return X.transpose(dim_1, dim_2).contiguous()


def generate_mask(seq_len):
    return torch.tril(torch.ones((1, 1, seq_len, seq_len), dtype=torch.bool))
