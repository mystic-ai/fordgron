import torch
from torch import nn
import torch.nn.functional as F
import math
from .rotary_embedding import RotaryEmbedding

class LinearSkipAddBias(nn.Module):
    def __init__(self, in_features: int, out_features: int, device=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty((out_features, in_features), device=device))
        self.bias = nn.Parameter(torch.empty(out_features, device=device))

    def forward(self, x):
        return F.linear(x, self.weight), self.bias

class SelfAttention(nn.Module):
    """
    Multi-head (if num_attention_heads > 1), scaled dot-product self-attention.
    """
    def __init__(self, embedding_dim, num_attention_heads, positional_encoding_implementation=None, rotary_pct=None, device=None):
        """
        Args:
            embedding_dim: int - length of the embedding vector
            num_attention_heads: int - number of attention heads
            mask: bool - whether to mask input triangularly
        """
        super().__init__()
        assert (
            embedding_dim % num_attention_heads == 0
        ), f"Embedding dimension ({embedding_dim}) must be divisible by number of heads ({num_attention_heads})"
        self.embedding_dim = embedding_dim
        self.num_attention_heads = num_attention_heads
        self.positional_encoding_implementation = positional_encoding_implementation
        self.embedding_dim_per_attention_head = self.embedding_dim // self.num_attention_heads
        self.num_rotary_dims = int(self.embedding_dim_per_attention_head * rotary_pct)
        self.to_queries_keys_values = nn.Linear(self.embedding_dim, 3 * self.embedding_dim)
        self.norm_factor = math.sqrt(self.embedding_dim_per_attention_head)
        self.rotary_embedding = RotaryEmbedding(num_rotary_dims=self.num_rotary_dims, device="meta") # figure out a way to make this conditional
        self.dense = LinearSkipAddBias(self.embedding_dim, self.embedding_dim, device=device)

    def forward(self, X, mask=True):
        """
        Args:
            X: torch.tensor [batch_len, seq_len, embedding_dim]
        Returns:
            attendedX: torch.tensor [batch_len, seq_len, embedding_dim]
        """
        batch_len, seq_len, input_embedding_dim = X.size()
        assert (
            input_embedding_dim == self.embedding_dim
        ), f"Input embedding dim ({input_embedding_dim}) must match layer embedding dim ({self.embedding_dim})"

        # queries, keys, and values introduce a learnable matrix on all the embedding vectors of the sequence
        queries_keys_values = self.to_queries_keys_values(X) # [batch_len, seq_len, (3 * embedding_dim)]
        
        queries_keys_values_across_heads = queries_keys_values.view(
            batch_len, seq_len, self.num_attention_heads, 3 * self.embedding_dim_per_attention_head
        ) # [batch_len, seq_len, num_attention_heads, (3 * embedding_dim_per_attention_head)]

        queries_layer = queries_keys_values_across_heads[..., :self.embedding_dim_per_attention_head] # [batch_len, seq_len, num_attention_heads, embedding_dim_per_attention_head] first chunk
        keys_layer = queries_keys_values_across_heads[..., self.embedding_dim_per_attention_head: 2 * self.embedding_dim_per_attention_head] # [batch_len, seq_len, num_attention_heads, embedding_dim_per_attention_head] second and middle chunk
        values_layer = queries_keys_values_across_heads[..., 2 * self.embedding_dim_per_attention_head:] # [batch_len, seq_len, num_attention_heads, embedding_dim_per_attention_head] third and last chunk

        if (self.positional_encoding_implementation == "rotary_embedding"):
            queries_layer, keys_layer = self.rotary_embedding(queries, keys, seq_len=query_seq_len) # both [batch_len, seq_len, num_attention_heads, embedding_dim_per_attention_head]

        # cache qkv values, add logic for this later

        # now compute attention!

        # we need to prepare for the dot product (batched matrix multiplication I guess)
        # TODO figure out tensor sizes here
        queries_layer = queries_layer.view(batch_len, seq_len * self.num_attention_heads, -1) # [batch_len, (seq_len * num_attention_heads), embedding_dim_per_attention_head]
        keys_layer = keys_layer.view(batch_len, seq_len * self.num_attention_heads, -1) # [batch_len, (seq_len * num_attention_heads), embedding_dim_per_attention_head]

        # preallocate attention score result tensor
        attention_scores = torch.empty(
            seq_len * self.num_attention_heads,
            batch_len,
            batch_len,
            device=queries_layer.device,
            dtype=queries_layer.dtype
        )

        # compute raw attention scores
        attention_scores = torch.baddbmm(
            attention_scores,
            queries_layer.transpose(0, 1), # [(seq_len * num_attention_heads), batch_len, embedding_dim_per_attention_head]
            keys_layer.transpose(0, 1).transpose(1, 2), # [(seq_len * num_attention_heads), embedding_dim_per_attention_head, batch_len]
            beta=0.0,
            alpha=(1.0 / self.norm_factor)
        ) # [(seq_len * num_attention_heads), batch_len, batch_len]

        attention_scores = attention_scores.view(seq_len, self.num_attention_heads, batch_len, batch_len) # [seq_len, num_attention_heads, batch_len, batch_len]

        # TODO get cached attention mask

        ltor_mask = torch.tril(
            torch.ones(
                (seq_len, batch_len, batch_len), device=attention_scores.device
            ).view(seq_len, 1, batch_len, batch_len).bool()
         ) # [seq_len, 1, batch_len, batch_len]

        if mask:
            attention_scores = attention_scores.masked_fill_(ltor_mask, -10000.0) # [seq_len, num_attention_heads, batch_len, batch_len]
            
        # should scale attention probs as well

        attention_probs = nn.Softmax(dim=-1)(attention_scores) # [seq_len, num_attention_heads, batch_len, batch_len]

        # attention dropout implementation should go here

        values_layer = values_layer.view(batch_len, seq_len * self.num_attention_heads, -1) # [batch_len, (seq_len * num_attention_heads), embedding_dim_per_attention_head]

        attention_probs = attention_probs.view(seq_len * self.num_attention_heads, batch_len, -1) # [(seq_len * num_attention_heads), batch_len, embedding_dim_per_attention_head]

        # here's the magic computation
        context_layer = torch.bmm(attention_probs, values_layer.transpose(0, 1)) # [(seq_len * num_attention_heads), batch_len, embedding_dim_per_attention_head]

        context_layer = context_layer.view(seq_len, self.num_attention_heads, batch_len, self.embedding_dim_per_attention_head) # [seq_len, num_attention_heads, batch_len, embedding_dim_per_attention_head]

        context_layer = context_layer.permute(2, 0, 1, 3).contiguous() # [batch_len, seq_len, num_attention_heads, embedding_dim_per_attention_head]

        context_layer = context_layer.view(batch_len, seq_len, self.embedding_dim) # [batch_len, seq_len, embedding_dim]

        output, bias = self.dense(context_layer)

        return output, bias
