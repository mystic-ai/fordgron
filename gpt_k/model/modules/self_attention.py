import torch
from torch import nn
import torch.nn.functional as F

class SelfAttention(nn.Module):
    def __init__(self, embedding_dim, num_heads, mask):
        super().__init__()
        assert (
            embedding_dim % num_heads == 0
        ), f"Embedding dimension ({embedding_dim}) should be divisible by nr. of heads ({num_heads})"
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.mask = mask
        self.toqueries = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.tokeys = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.tovalues = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.unifyheads = nn.Linear(embedding_dim, embedding_dim)

    def forward(self, x):
        batch_len, seq_len, input_embedding_dim = x.size()
        assert (
            input_embedding_dim == self.embedding_dim
        ), f"Input embedding dim ({input_embedding_dim}) should match layer embedding dim ({self.embedding_dim})"

        num_layers = self.embedding_dim // self.num_heads

        queries = self.toqueries(x)
        keys = self.tokeys(x)
        values = self.tovalues(x)

        queries = queries.view(batch_len, seq_len, self.num_heads, num_layers)
        keys = keys.view(batch_len, seq_len, self.num_heads, num_layers)
        values = values.view(batch_len, seq_len, self.num_heads, num_layers)

        queries = (
            queries.transpose(1, 2)
            .contiguous()
            .view(batch_len * self.num_heads, seq_len, num_layers)
        )
        keys = (
            keys.transpose(1, 2)
            .contiguous()
            .view(batch_len * self.num_heads, seq_len, num_layers)
        )
        values = (
            values.transpose(1, 2)
            .contiguous()
            .view(batch_len * self.num_heads, seq_len, num_layers)
        )

        dot = torch.bmm(queries, keys.transpose(1, 2)) / torch.sqrt(keys.size()[-1])
        assert dot.size() == (batch_len * self.num_heads, seq_len, seq_len)
        if self.mask:
            height, width = dot.size(-2), dot.size(-1)
            indices = torch.triu_indices(height, width)
            dot[..., indices[0], indices[1]] = float("-inf")
           
        dot = F.softmax(dot, dim=2)
        out = torch.bmm(dot, values).view(
            batch_len, self.num_heads, seq_len, num_layers
        )
        out = (
            out.transpose(1, 2)
            .contiguous()
            .view(batch_len, seq_len, num_layers * self.num_heads)
        )
        return self.unifyheads(out)