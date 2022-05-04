import torch.nn as nn
import torch
import math

from .modules import TransformerBlock


class Transformer(nn.Module):
    def __init__(self, args, device=None):
        super().__init__()

        """
        if use_cache is True, then the keys and values from the previous transformer block are saved and sent to the next transformer block
        by preventing this extra computation, inference can be a little faster at the cost of model performance (I believe, although there seems to be 0 research on this)
        """
        self.swap_batch_len_and_seq_len = args["swap_batch_len_and_seq_len"]

        """
        embedding layer which acts as a fancy lookup table, projecting the token id (integer) to a vector with dimension embedding_dim
        since this embedding vector is trained with the model, it has learned 'representations' of that token by how it relates to others, aka 'the meaning of' that token
        """
        self.token_id_embedding = nn.Embedding(args["vocab_len"], args["embedding_dim"], device=device)
        self.positional_encoding = nn.Embedding(args["seq_len"], args["embedding_dim"], device=device)

        """
        the core of the transformer is a stack of transformer blocks, each of which essentially applies self-attention (and a few other tricks) to the output of the previous block
        atm this is a decoder-only architecture like GPT-2, but eventually it will be extended to include encoders
        the first block receives the 'embedded' tokens, and each block ouptuts a tensor with that same shape
        """
        self.decoder_stack = nn.ModuleList([])
        for layer_i in range(args["depth"]):
            self.decoder_stack.append(TransformerBlock(args, device=device))

        """
        after the final transformer block there is an additional layer norm (pattern set by GPT-2)
        supposedly this creates norm invariance without requiring weight standardisation (in other words, it makes training more stable)
        """
        self.final_layer_norm = nn.LayerNorm(
            args["embedding_dim"],
            eps=args["layernorm_eps"],
            device=device,
        )

        """
        finally the model needs to 'reverse the embedding' by projecting the hidden states of attended tokens out into a vector with dimension vocab_len
        at this stage the model outputs logits which we can sample in order to select the next token in the generation process, bearing in mind that the next token must be in the vocab (and therefore its token id will be represented by one item in the vocab_len vector)
        """
        self.logits = nn.Linear(
            args["embedding_dim"],
            args["vocab_len"],
            bias=False, # True on J, False on 20B
            device=device,
        )

    def forward(self, X, attention_mask=None):
        """
        for every new token that requires generation, the forward function will be called on the transformer
        this is because one-token-at-a-time generation is setup, and the state of the transformer needs to be reset (bar caching) on each new token request
        it would be interesting to explore how two-tokens-at-a-time generation would behave

        X: torch.Tensor [batch_len, input_seq_len] int = token ids from the tokenizer
        attention_mask: torch.Tensor [?] bool = shows 'True' at the indices of the 'visible' token id and 'False' at the indices of the 'masked' token id
        layer_past: torch.Tensor [?] ? = attention keys and values from the previous transformer block, so they can be reused 
        """

        """
        setup is complete, now we need to pass the inputs through the transformer. see the init function for details on what each module does
        X = [batch_len, input_seq_len]
        0.000_025_033950805664062 s
        """
        embedded_tokens = self.token_id_embedding(X) # [batch_len, input_seq_len, embedding_dim]
        position_ids = torch.arange(0, X.size(-1), dtype=torch.long, device=X.device)

        hidden_states = torch.add(embedded_tokens, self.positional_encoding(position_ids))

        """
        atm a new mask is built on each forward pass, although it is likely possible that they can be cached (but this should be benchmarked)
        attention_mask is a tensor of Trues and Falses where the indices of all masked tokens are False
        0.000_028_371810913085938 s
        """
        if attention_mask is None:
            attention_mask = make_causal_mask(X.size(1), swap_batch_len_and_seq_len=self.swap_batch_len_and_seq_len).to(X.device)

        """
        some models were trained with batch_len and seq_len swapped, so during inference it's necessary to perform the swap in order to make the weights work
        0.000_006_198883056640625 s
        """
        if self.swap_batch_len_and_seq_len:
            hidden_states = self.swap_dimensions(hidden_states, 0, 1) # [input_seq_len, batch_len, embedding_dim]
    
        """
        we must manually pass the hidden states through each layer of the stack in order to set the true layer_past and extract the kv_cache
        0.029_156_68487548828 s
        """
        for layer_i, transformer_block in enumerate(self.decoder_stack):
            hidden_states = transformer_block(
                hidden_states,
                attention_mask=attention_mask,
            ) # [input_seq_len, batch_len, embedding_dim]

        """

        reverse the dimension swap if necessary
        0.000_005_0067901611328125 s
        """
        if self.swap_batch_len_and_seq_len:
            hidden_states = self.swap_dimensions(hidden_states, 0, 1) # [input_seq_len, batch_len, embedding_dim]

        """
        pass through a final layer norm
        0.000_381_70814514160156 s
        """
        hidden_states = self.final_layer_norm(hidden_states) # [batch_len, input_seq_len, embedding_dim]

        """
        0.000_022_88818359375
        """
        logits = self.logits(hidden_states) # [batch_len, input_seq_len, vocab_len]

        """
        only return the logits if kv_caching isn't on
        """
        return logits

    @classmethod
    def swap_dimensions(cls, X, dim_1, dim_2):
        return X.transpose(dim_1, dim_2).contiguous()

def make_causal_mask(seq_len, swap_batch_len_and_seq_len=False):
    # might need to put these on GPU
    if swap_batch_len_and_seq_len:
        return torch.tril(torch.ones((1, 1, seq_len, seq_len), dtype=torch.bool))
    else:
        return torch.tril(torch.ones((seq_len, seq_len), dtype=torch.bool))
