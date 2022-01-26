from torch import nn
import torch.nn.functional as F
from .self_attention import SelfAttention

class TransformerBlock(nn.Module):
    """
    Attributes:
        attention: SelfAttention - scaled multi-head attention layer
        norm1: nn.LayerNorm
        ff: nn.Sequential - feedforward network
        norm2: nn.LayerNorm
        dropout: nn.Dropout
    """
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

    def forward(self, X):
        """
        The size of X remains the same as it passes through the transformer block. 
        
        Args:
            X: torch.tensor [batch_size, seq_len, embedding_dim]
        """
        attended = self.attention(X)
        X = self.norm1(attended + X)
        X = self.dropout(X)
        fedforward = self.ff(X)
        X = self.norm2(fedforward + X)
        X = self.dropout(X)
        return X