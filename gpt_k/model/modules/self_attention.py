import torch
from torch import nn
import torch.nn.functional as F
import math
from .rotary_embedding import RotaryEmbedding

class SelfAttention(nn.Module):
    """
    Multi-head (if num_attention_heads > 1), scaled dot-product self-attention.
    """
    def __init__(self, embedding_dim, num_attention_heads, positional_encoding_implementation=None, rotary_pct=None):
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
        self.dense = nn.Linear(self.embedding_dim, self.embedding_dim)

    def forward(self, X, mask=True):
        """
        Args:
            X: torch.tensor [batch_len, seq_len, input_embedding_dim]
        Returns:
            attendedX: torch.tensor [batch_len, seq_len, embedding_dim]
        """
        batch_len, seq_len, input_embedding_dim = X.size()
        assert (
            input_embedding_dim == self.embedding_dim
        ), f"Input embedding dim ({input_embedding_dim}) must match layer embedding dim ({self.embedding_dim})"

        # queries, keys, and values introduce a learnable matrix on all the embedding vectors of the sequence
        # typically [batch_len, seq_len, embedding_dim] -> [batch_len, seq_len, (3 * embedding_dim)]
        queries_keys_values = self.to_queries_keys_values(X)
        
        query_seq_len = self.embedding_dim // self.num_attention_heads

        # 'split' queries keys and values across heads by reshaping so that num_attention_heads is a thing
        # we're reshaping by using the embedding_dim_per_attention_head
        # typically [batch_len, seq_len, (3 * embedding_dim)] -> [batch_len, seq_len, num_attention_heads, (3 * embedding_dim_per_attention_head)]
        queries_keys_values_across_heads = qkv.view(
            qkv.size()[:-1] + (
                self.num_attention_heads, 3 * self.embedding_dim_per_attention_head
            )
        )

        # get the three 'elements' of qkv
        queries_layer = queries_keys_values_across_heads[..., :self.embedding_dim_per_attention_head] # first chunk (of most nested tensor bit)
        keys_layer = queries_keys_values_across_heads[..., self:embedding_dim_per_attention_head: 2 * self.embedding_dim_per_attention_head] # second and middle chunk (of most nested tensor bit)
        values_layer = queries_keys_values_across_heads[..., 2 * self.embedding_dim_per_attention_head:] # third and last chunk (of most nested tensor bit)

        if (self.positional_encoding_implementation == "rotary_embedding"):
            queries_layer, keys_layer = self.rotary_embedding(queries, keys, seq_len=query_seq_len)

        # cache qkv values, add logic for this later

        # now compute attention!

        # queries_layer is in this shape [batch_len, seq_len, num_attention_heads, embedding_dim_per_attention_head]
        # we need to prepare for the dot product (batched matrix multiplication I guess)
        # TODO figure out tensor sizes here
        queries_layer = queries_layer.view(queries_layer.size(0), queries_layer.size(1) * queries_layer.size(2), -1)
        keys_layer = keys_layer.view(keys_layer.size(0), queries_layer.size(1) * queries_layer.size(2), -1)

        # preallocate attention score result tensor
        attention_scores = torch.empty(
            queries_layer.size(1) * queries_layer.size(2),
            queries_layer.size(0),
            keys_layer.size(0),
            dtype=queries_layer.dtype,
            device=queries_layer.device 
        )

        # compute raw attention scores
        attention_scores = torch.baddbmm(
            attention_scores,
            queries_layer.transpose(0, 1),
            keys_layer.transpose(0, 1).transpose(1, 2),
            beta=0.0,
            alpha=(1.0 / self.norm_factor)
        )

        # TODO figure out tensor size
        attention_scores = attention_scores.view(queries_layer.size(1), queries_layer.size(2), queries_layer.size(0), keys_layer.size(0))

        # TODO get cached attention mask

        if self.mask:
            attention_scores = attention_scores.masked_fill_(mask, -10000.0)
           
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # attention dropout implementation should go here

        context_layer_shape = (
            values_layer.size(1),
            values_layer.size(2),
            queries_layer.size(0),
            values_layer.size(3)
        )

        values_layer = values_layer.view(values_layer.size(0), context_layer_shape[0] * context_layer_shape[1], -1)

        attention_probs = attention_probs.view(context_layer_shape[0] * context_layer_shape[1], context_layer_shape[2], -1)

        context_layer = torch.bmm(attention_probs, values_layer.transpose(0, 1))

        context_layer = context_layer.permute(2, 0, 1, 3).contiguous()

        new_context_layer_shape = context_layer.size()[:2] + (self.embedding_dim)

        context_layer = context_layer.view(*new_context_layer_shape)

        output, bias = self.dense(context_layer)

        return output, bias
    
    def _split_data_across_heads(self, A, num_attention_heads, query_len):
        """
        Divide the batched data across num_attention_heads. The split occurs 'in theory' by reshaping the matrix, rather than slicing it.
        
        Args:
            A: torch.tensor - [batch_len, seq_len, embedding_dim]
            num_attention_heads = int - number of attention heads
            query_len = int - chunk of embedding vector per head
        Returns:
            A: torch.tensor - [batch_len * num_attention_heads, seq_len, query_len]
        """
        batch_len, seq_len, embedding_dim = A.size()
        # embedding vectors inside the queries, keys, and values are divided into heads and sublayers of query_len length
        A = A.view(batch_len, seq_len, num_attention_heads, query_len)
        A = A.transpose(1, 2).contiguous() # swap seq_len and num_attention_heads dimensions, to enable next operation
        A = A.view(batch_len * num_attention_heads, seq_len, query_len)
        return A