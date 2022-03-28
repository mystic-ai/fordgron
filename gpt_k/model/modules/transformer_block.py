from torch import nn
import torch.nn.functional as F
from .self_attention import SelfAttention
from .feed_forward import FeedForward

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
        num_attention_heads,
        mask,
        dropout,
        forward_expansion,
        positional_encoding_implementation,
        rotary_pct,
        layernorm_epsilon
    ):
        super().__init__()
        self.attention = SelfAttention(embedding_dim, num_attention_heads, positional_encoding_implementation=positional_encoding_implementation, rotary_pct=rotary_pct)
        self.norm1 = nn.LayerNorm(embedding_dim, eps=layernorm_epsilon)
        self.feedforward = FeedForward(embedding_dim)
        self.norm2 = nn.LayerNorm(embedding_dim, eps=layernorm_epsilon)
        self.dropout = nn.Dropout(dropout)
        self.mask = mask

    def forward(self, X):
        """
        The size of X remains the same as it passes through the transformer block. 
        
        Args:
            X: torch.tensor [batch_size, seq_len, embedding_dim]
        """
        residual = X
        norm1X = self.norm1(X)
        attendedX, attendedX_bias, _ = self.attention(norm1X, self.mask)
        attention_output = attendedX + attendedX_bias.expand_as(attendedX)
        norm2X = self.norm2(X)
        fedforwardX, fedforwardX_bias = self.feedforward(norm2X)
        fedforward_output = fedforwardX + fedforwardX_bias.expand_as(fedforwardX)
        output = residual + fedforward_output
        return output