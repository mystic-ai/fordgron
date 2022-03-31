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
        self.use_cache = True

    def forward(self, X, mask=None, previous_layer=None):
        """
        Args:
            X: torch.tensor [batch_len, input_seq_len, embedding_dim]
        Returns:
            attendedX: torch.tensor [batch_len, input_seq_len, embedding_dim]
        """
        # previous_layer is [2, input_seq_len_so_far, batch_len, num_attention_heads, embedding_dim_per_attention_head] the 2 is keys and values

        # queries, keys, and values introduce a learnable matrix on all the embedding vectors of the sequence
        queries_keys_values = self.to_queries_keys_values(X) # [batch_len, input_seq_len, (3 * embedding_dim)]

        new_tensor_shape = queries_keys_values.size()[:-1] + (
            self.num_attention_heads,
            3 * self.embedding_dim_per_attention_head,
        )
        queries_keys_values_across_heads = queries_keys_values.view(*new_tensor_shape)

        queries_layer = queries_keys_values_across_heads[..., :self.embedding_dim_per_attention_head] # [batch_len, input_seq_len, num_attention_heads, embedding_dim_per_attention_head] first chunk
        keys_layer = queries_keys_values_across_heads[..., self.embedding_dim_per_attention_head: 2 * self.embedding_dim_per_attention_head] # [batch_len, input_seq_len, num_attention_heads, embedding_dim_per_attention_head] second and middle chunk
        values_layer = queries_keys_values_across_heads[..., 2 * self.embedding_dim_per_attention_head:] # [batch_len, input_seq_len, num_attention_heads, embedding_dim_per_attention_head] third and last chunk

        if (self.positional_encoding_implementation == "rotary_embedding"):
            queries_layer, keys_layer = self.rotary_embedding(queries_layer, keys_layer, values_layer, previous_layer=previous_layer) # both [batch_len, input_seq_len, num_attention_heads, embedding_dim_per_attention_head]

        if previous_layer is not None and previous_layer.numel() > 0:
            past_key, past_value = previous_layer # extracting keys and values from one larger tensor, both are [batch_len, seq_len, num_attention_heads, embedding_dim_per_attention_head]
            print("keys layer before stacking")
            print(keys_layer.size())
            keys_layer = torch.cat((past_key.type_as(keys_layer), keys_layer), dim=0) # they are stacked, so now [batch_len, seq_len, num_attention_heads, embedding_dim_per_attention_head]
            print("keys layer after stacking")
            print(keys_layer.size())
            # I think the dimension on the cating depends on swapping of seq len and batch len
            values_layer = torch.cat((past_value.type_as(values_layer), values_layer), dim=0) # they are stacked, so now [batch_len, seq_len, num_attention_heads, embedding_dim_per_attention_head]

        # caching keys and values before computing attention

        if self.use_cache:
            kv_cache = torch.stack((keys_layer, values_layer)) # [2, batch_len, input_seq_len, num_attention_heads, embedding_dim_per_attention_head]
        else:
            print("use_cache is false")
            kv_cache = None

        # now compute attention!

        output_size = (
            queries_layer.size(1),
            queries_layer.size(2),
            queries_layer.size(0),
            keys_layer.size(0)
        )

        # we need to prepare for the dot product (batched matrix multiplication I guess)
        queries_layer = queries_layer.view(output_size[2], output_size[0] * output_size[1], -1) # [batch_len, (input_seq_len * num_attention_heads), embedding_dim_per_attention_head]
        print("queries layer size")
        print(queries_layer.size())
        keys_layer = keys_layer.view(output_size[3], output_size[0] * output_size[1], -1) # [batch_len, (input_seq_len * num_attention_heads), embedding_dim_per_attention_head]
        print("keys layer size")
        print(queries_layer.size())

        # preallocate attention score result tensor
        attention_scores = torch.empty(
            output_size[0] * output_size[1],
            output_size[2],
            output_size[3],
            device=queries_layer.device,
            dtype=queries_layer.dtype
        )

        # compute raw attention scores
        attention_scores = torch.baddbmm(
            attention_scores,
            queries_layer.transpose(0, 1), # [(input_seq_len * num_attention_heads), batch_len, embedding_dim_per_attention_head]
            keys_layer.transpose(0, 1).transpose(1, 2), # [(input_seq_len * num_attention_heads), batch_len, embedding_dim_per_attention_head]
            beta=0.0,
            alpha=(1.0 / self.norm_factor)
        ) # [input_seq_len * num_attention_heads, batch_len, batch_len]

        print("attention scores")
        print(attention_scores.size())

        # first time [128, 12, 12]
        # second time [128, 1, 13] OKAY HERE'S THE PROBLEM, seq_len is not the updated seq_len at some point

        attention_scores = attention_scores.view(*output_size) # [input_seq_len, num_attention_heads, input_seq_len, batch_len]

        print("viewed attention scores")
        print(attention_scores.size())
        
        
        # first time [2, 64, 12, 12]
        # second time [2, 64, 1, 13] WHY IS THIS 1 AND NOT 13

        if self.use_cache:
            mask = mask[..., :attention_scores.size(3), :attention_scores.size(3)]

        masked_scores = attention_scores.masked_fill_(mask, -10000.0) # [seq_len, num_attention_heads, batch_len, batch_len]
            
        # should scale attention probs as well

        attention_probs = nn.Softmax(dim=-1)(masked_scores) # [seq_len, num_attention_heads, batch_len, batch_len]

        # attention dropout implementation should go here

        output_size = (
            values_layer.size(1),
            values_layer.size(2),
            queries_layer.size(0),
            values_layer.size(3),
        ) # [seq_len, num_attention_heads, batch_len, embedding_dim_per_attention_head]

        values_layer = values_layer.view(values_layer.size(0), output_size[0] * output_size[1], -1) # [batch_len, (seq_len * num_attention_heads), embedding_dim_per_attention_head]

        attention_probs = attention_probs.view(output_size[0] * output_size[1], output_size[2], -1) # [(seq_len * num_attention_heads), batch_len, embedding_dim_per_attention_head]

        # here's the magic computation
        context_layer = torch.bmm(attention_probs, values_layer.transpose(0, 1))

        context_layer = context_layer.view(*output_size)

        context_layer = context_layer.permute(2, 0, 1, 3).contiguous() # [batch_len, input_seq_len, num_attention_heads, embedding_dim_per_attention_head]

        new_context_layer_shape = context_layer.size()[:-2] + (
            self.embedding_dim,
        ) # [batch_len, input_seq_len, embedding_dim]
        context_layer = context_layer.view(*new_context_layer_shape)

        output, bias = self.dense(context_layer)

        if self.use_cache:
            output = [output, kv_cache]

        return output, bias
