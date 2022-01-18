import torch
from torch import nn
import torch.nn.functional as F

from .modules import TransformerBlock, PositionalEmbedding


class Transformer(nn.Module):
    def __init__(
        self,
        embedding_dim,
        num_heads,
        mask,
        dropout,
        forward_expansion,
        depth,
        seq_len,
        vocab_len,
        device,
    ):
        super().__init__()
        self.device = device
        self.vocab_len = vocab_len
        self.embedding = nn.Embedding(vocab_len, embedding_dim)
        self.positional_embedding = PositionalEmbedding(embedding_dim, 0, seq_len)

        modules = [
            TransformerBlock(
                embedding_dim,
                num_heads,
                mask,
                dropout,
                forward_expansion,
            )
            for _ in range(depth)
        ]
        self.model = nn.Sequential(*modules)
        self.output_probs = nn.Linear(embedding_dim, vocab_len)

    def forward(self, input):
        tokens = self.embedding(input)
        batch_len, seq_len, embedding_dim = tokens.size()
        positions = self.positional_embedding(tokens)
        x = self.model(positions)
        y = self.output_probs(positions.view(batch_len * seq_len, embedding_dim)).view(batch_len, seq_len, self.vocab_len)
        return F.log_softmax(y, dim=2)
