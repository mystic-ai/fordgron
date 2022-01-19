from torch import nn
import torch.nn.functional as F
from self_attention import SelfAttention

class TransformerBlock(nn.Module):
    def __init__(
        self,
        embedding_dim,
        num_heads,
        mask,
        dropout,
        forward_expansion,
    ):
        super().__init__()
        self.attention = SelfAttention(embedding_dim, num_heads, mask)
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.ff = nn.Sequential(
            nn.Linear(embedding_dim, forward_expansion * embedding_dim),
            nn.ReLU(),
            nn.Linear(forward_expansion * embedding_dim, embedding_dim),
        )
        self.norm2 = nn.LayerNorm(embedding_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        attended = self.attention(x)
        x = self.norm1(attended + x)
        x = self.dropout(x)
        fedforward = self.ff(x)
        x = self.norm2(fedforward + x)
        x = self.dropout(x)
        return x