import torch
from torch import nn
import torch.nn.functional as F

from .modules import TransformerBlock


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
        self.token_embedding = nn.Embedding(embedding_dim, vocab_len)
        self.positional_embedding = nn.Embedding(seq_len, embedding_dim)

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

    def forward(self, x):
        tokens = self.token_embedding(x)
        b, t, e = tokens.size()
        positions = self.positional_embedding(torch.arange(t, device=self.device))[
            None, :, :
        ].expand(b, t, e)
        x = tokens + positions
        x = self.model(x)
        y = self.output_probs(x.view(b * t, e)).view(b, t, self.vocab_len)
        return F.log_softmax(y, dim=2)
