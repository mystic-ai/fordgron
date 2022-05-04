import torch
import math
from torch import nn, Tensor
from .rotary_embedding import RotaryEmbedding, apply_rotary_pos_emb
import math

class SelfAttention(nn.Module):
    def __init__(self, args, device=None):
        super().__init__()
        self.embedding_dim = args["embedding_dim"]
        self.num_attention_heads = args["num_attention_heads"]
        self.embedding_dim_per_attention_head = self.embedding_dim // self.num_attention_heads
        self.to_query_key_value = nn.Linear(self.embedding_dim, 3 * self.embedding_dim)
        self.dense = nn.Linear(self.embedding_dim, self.embedding_dim)
        self.softmax = nn.Softmax(dim=-1)
        self.register_buffer("masked_bias", torch.tensor(args["masked_bias"]))

    def _split_attention_heads(self, X: Tensor):
        batch_len, seq_len, _ = X.size()
        X = X.view(batch_len, seq_len, self.num_attention_heads, self.embedding_dim_per_attention_head) # [batch_len, seq_len, num_attention_heads, embedding_dim_per_attention_head]
        return X.transpose(1, 2) # [batch_len, num_attention_heads, seq_len, embedding_dim_per_attention_head]

    def _merge_attention_heads(self, X: Tensor):
        batch_len, num_attention_heads, seq_len, embedding_dim_per_attention_head = X.size()
        return X.transpose(1, 2).contiguous().view(batch_len, seq_len, self.embedding_dim)

    def multihead_attention(self, query: Tensor, key: Tensor, value: Tensor, attention_mask: Tensor):
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.embedding_dim_per_attention_head)
        if attention_mask is not None:
            scores = scores.masked_fill(attention_mask, -1e-4)
        scores = self.softmax(scores)
        a = torch.matmul(scores, value)
        return a

    def forward(self, X: Tensor, attention_mask: Tensor, layer_past=None):
        """
        X: [batch_len, seq_len, embedding_dim]
        """
        X = self.to_query_key_value(X) # [batch_len, seq_len, 3 * embedding_dim])
        query, key, value = X.split(self.embedding_dim, dim=-1) # each [batch_len, seq_len, embedding_dim]
        query, key, value = map(self._split_attention_heads, (query, key, value)) # each [batch_len, num_attention_heads, seq_len, embedding_dim_per_attention_head]
        X = self.multihead_attention(query, key, value, attention_mask)
        X = self._merge_attention_heads(X)
        X = self.dense(X)
        return X