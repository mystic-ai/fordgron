import torch
from torch import nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding with alternating sin / cos. Dropout as well.
    
    Attributes:
        P: torch.tensor [1, seq_len, embedding_dim] 
        dropout: nn.Dropout
    """
    def __init__(self, dropout, seq_len, embedding_dim):
        super().__init__()
        self.P = torch.zeros([1, seq_len, embedding_dim])
        X = torch.arange(seq_len) # vector of indices up to seq_len
        X = X.view(-1, 1) # unsqueeze all elements of the vector individually [seq_len, 1]
        # part one of the positional encoding, batched across the sequence length rather than looped
        untitled_var = torch.pow(10000, torch.arange(0, embedding_dim, 2) / embedding_dim) # [512]
        X = X / untitled_var # [seq_len, pe_len]
        # part two of the positional encoding
        self.P[:, :, 0::2] = torch.sin(X) # sin alternating from index 0 (even)
        self.P[:, :, 1::2] = torch.cos(X) # cos alternating from index 1 (odd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, X):
        """
        Sums the positional embedding with the embeddings, and applies dropout.
        
        Args:
            X: torch.tensor [batch_size, seq_len, embedding_dim] - embedding vectors
        Returns:
            X: torch.tensor [batch_size, seq_len, embedding_dim] - embedding vectors with positional information added
        """
        seq_len = X.size()[1]
        X = X + self.P[:, :seq_len, :].to(X.device) # from P add to all batches, rows up to seq_len, all embeddings
        X = self.dropout(X)
        return X