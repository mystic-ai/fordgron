import torch
from torch import nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    def __init__(self, embedding_dim, dropout, seq_len):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.P = torch.zeros([1, seq_len, embedding_dim])
        X = torch.arange(seq_len, dtype=torch.float32).reshape(
            -1, 1) / torch.pow(10000, torch.arange(
            0, embedding_dim, 2, dtype=torch.float32) / embedding_dim)
        self.P[:, :, 0::2] = torch.sin(X)
        self.P[:, :, 1::2] = torch.cos(X)

    def forward(self, X):
        X = X + self.P[:, :X.shape[1], :].to(X.device)
        return self.dropout(X)