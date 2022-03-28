import torch
from torch import nn
import torch.nn.functional as F
import math
from rotary_embedding import RotaryEmbedding

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
        self.to_queries_keys_values = nn.Linear(self.embedding_dim, 3 * self.embedding_dim, bias=False)
        self.norm_factor = math.sqrt(self.embedding_dim_per_attention_head)
        if (positional_encoding_implementation == "rotary_embedding"):
            self.rotary_embedding = RotaryEmbedding(num_rotary_dims=self.num_rotary_dims, device="meta")

    def forward(self, X):
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
        # typically [batch_len, seq_len, (3 * embedding_dim)] -> [batch_len, seq_len, num_attention_heads, (3 * embedding_dim)]
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
            

        
        # attention operation: what is the 'relevance' between queries and keys?
        dot = torch.bmm(queries, keys.transpose(1, 2)) # transpose on keys is likely necessary
        assert dot.size() == (batch_len * self.num_attention_heads, seq_len, seq_len)
        
        if self.mask:
            height, width = dot.size(-2), dot.size(-1)
            indices = torch.triu_indices(height, width)
            dot[..., indices[0], indices[1]] = float("-1e20")
           
        dot = F.softmax(dot, dim=2)
        out = torch.bmm(dot, values).view(
            batch_len, self.num_attention_heads, seq_len, query_len
        )
        out = (
            out.transpose(1, 2)
            .contiguous()
            .view(batch_len, seq_len, query_len * self.num_attention_heads)
        )
        
        attendedX = self.unify_heads(out)
    
        return attendedX
    
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