from .util import mask_, d, slice_diag

import torch
from torch import nn
import torch.nn.functional as F

import random, math, sys

class SelfAttention(nn.Module):
    def __init__(self, embedding_dim, num_heads=8, mask=False):
        super().__init__()
        assert embedding_dim % num_heads == 0, f"Embedding dimension ({embedding_dim}) should be divisible by the number of heads ({num_heads})"

        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.mask = mask

        self.tokeys = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.toqueries = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.tovalues = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.unifyheads = nn.Linear(embedding_dim, embedding_dim)

    def forward(self, x):

        b, t, e = x.size()
        assert e == self.embedding_dim, f'Input embedding dim ({e}) should match layer embedding dim ({self.embedding_dim})'

        s = e // self.num_heads

        keys    = self.tokeys(x)
        queries = self.toqueries(x)
        values  = self.tovalues(x)

        keys    = keys.view(b, t, self.num_heads, s)
        queries = queries.view(b, t, self.num_heads, s)
        values  = values.view(b, t, self.num_heads, s)

        # -- We first compute the k/q/v's on the whole embedding vectors, and then split into the different heads.
        #    See the following video for an explanation: https://youtu.be/KmAISyVvE1Y

        # Compute scaled dot-product self-attention

        # - fold heads into the batch dimension
        keys = keys.transpose(1, 2).contiguous().view(b * h, t, s)
        queries = queries.transpose(1, 2).contiguous().view(b * h, t, s)
        values = values.transpose(1, 2).contiguous().view(b * h, t, s)

        queries = queries / (e ** (1/4))
        keys    = keys / (e ** (1/4))
        # - Instead of dividing the dot products by sqrt(e), we scale the keys and values.
        #   This should be more memory efficient

        # - get dot product of queries and keys, and scale
        dot = torch.bmm(queries, keys.transpose(1, 2))

        assert dot.size() == (b*h, t, t)

        if self.mask: # mask out the upper half of the dot matrix, excluding the diagonal
            mask_(dot, maskval=float('-inf'), mask_diagonal=False)

        dot = F.softmax(dot, dim=2)
        # - dot now has row-wise self-attention probabilities

        # apply the self attention to the values
        out = torch.bmm(dot, values).view(b, h, t, s)

        # swap h, t back, unify heads
        out = out.transpose(1, 2).contiguous().view(b, t, s * h)

        return self.unifyheads(out)

class TransformerBlock(nn.Module):

    def __init__(self, emb, heads, mask, seq_length, ff_hidden_mult=4, dropout=0.0, attention_type='default', pos_embedding=None):
        super().__init__()
        if attention_type == 'default':
            self.attention = SelfAttention(emb, num_heads=heads, mask=mask)
        else:
            raise Exception(f'Self-attention type {type} not recognized.')

        self.mask = mask

        self.norm1 = nn.LayerNorm(emb)
        self.norm2 = nn.LayerNorm(emb)

        self.ff = nn.Sequential(

            nn.Linear(emb, ff_hidden_mult * emb),
            nn.ReLU(),
            nn.Linear(ff_hidden_mult * emb, emb)
        )

        self.do = nn.Dropout(dropout)

    def forward(self, x):

        attended = self.attention(x)

        x = self.norm1(attended + x)

        x = self.do(x)

        fedforward = self.ff(x)

        x = self.norm2(fedforward + x)

        x = self.do(x)

        return x