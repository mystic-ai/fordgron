import torch
from torch import nn
import torch.nn.functional as F

class SelfAttention(nn.Module):
    """
    Multi-head (if num_heads > 1), scaled dot-product self-attention.
    """
    def __init__(self, embedding_dim, num_heads, mask):
        """
        Args:
            embedding_dim: int - length of the embedding vector
            num_heads: int - number of attention heads
            mask: bool - whether to mask input triangularly
        """
        super().__init__()
        assert (
            embedding_dim % num_heads == 0
        ), f"Embedding dimension ({embedding_dim}) must be divisible by number of heads ({num_heads})"
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.mask = mask
        self.to_queries = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.to_keys = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.to_values = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.unify_heads = nn.Linear(embedding_dim, embedding_dim)

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

        # queries, keys, and values will retain the shape of X. they introduce a learnable matrix on all the embedding vectors of the sequence
        queries = self.to_queries(X)
        keys = self.to_keys(X)
        values = self.to_values(X)
        
        query_len = self.embedding_dim // self.num_heads

        queries = self._split_data_across_heads(queries, num_heads=self.num_heads, query_len=query_len)
        keys = self._split_data_across_heads(keys, num_heads=self.num_heads, query_len=query_len)
        values = self._split_data_across_heads(values, num_heads=self.num_heads, query_len=query_len)

        # scale queries and keys before matmul (done early for memory efficiency)
        queries = queries / (self.embedding_dim ** (1/4))
        keys = keys / (self.embedding_dim ** (1/4))
        
        # attention operation: what is the 'relevance' between queries and keys?
        dot = torch.bmm(queries, keys.transpose(1, 2)) # transpose on keys is likely necessary
        assert dot.size() == (batch_len * self.num_heads, seq_len, seq_len)
        
        if self.mask:
            height, width = dot.size(-2), dot.size(-1)
            indices = torch.triu_indices(height, width)
            dot[..., indices[0], indices[1]] = float("-1e20")
           
        dot = F.softmax(dot, dim=2)
        out = torch.bmm(dot, values).view(
            batch_len, self.num_heads, seq_len, query_len
        )
        out = (
            out.transpose(1, 2)
            .contiguous()
            .view(batch_len, seq_len, query_len * self.num_heads)
        )
        
        attendedX = self.unify_heads(out)
    
        return attendedX
    
    def _split_data_across_heads(self, A, num_heads, query_len):
        """
        Divide the batched data across num_heads. The split occurs 'in theory' by reshaping the matrix, rather than slicing it.
        
        Args:
            A: torch.tensor - [batch_len, seq_len, embedding_dim]
            num_heads = int - number of attention heads
            query_len = int - chunk of embedding vector per head
        Returns:
            A: torch.tensor - [batch_len * num_heads, seq_len, query_len]
        """
        batch_len, seq_len, embedding_dim = A.size()
        # embedding vectors inside the queries, keys, and values are divided into heads and sublayers of query_len length
        A = A.view(batch_len, seq_len, num_heads, query_len)
        A = A.transpose(1, 2).contiguous() # swap seq_len and num_heads dimensions, to enable next operation
        A = A.view(batch_len * num_heads, seq_len, query_len)
        return A