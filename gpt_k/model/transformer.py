from torch import nn
import torch.nn.functional as F

from .modules import PositionalEncoding, TransformerBlock


class Transformer(nn.Module):
    """
    Decoder-only GPT-style transformer.
    
    Uses positional encoding instead of a learned embedding. Forward pass takes in a raw list of tokens – with no fixed sequence length – and returns softmaxed probabilities for the next word.
    
    Attributes:
        vocab_len: int - size of entire vocabulary
        token_embedding: nn.Embedding - embedding layer which converts token sequences to token embeddings
        positional_encoding: PositionalEncoding - injects sequential position information into the token embeddings
        model: nn.Sequential - stack of transformer blocks which outputs the attended matrix
        output: nn.Linear - converts attended matrix to logits across entire vocab_len
    """
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
    ):
        super().__init__()
        self.vocab_len = vocab_len
        self.token_embedding = nn.Embedding(vocab_len, embedding_dim)
        self.positional_encoding = PositionalEncoding(dropout, seq_len, embedding_dim)

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
        self.output = nn.Linear(embedding_dim, vocab_len) # given predicted token embeddings, what's the next word?

    def forward(self, input):
        """
        Args:
            input: torch.tensor [batch_len, seq_len] - matrix of token sequences
        Returns:
            softmaxed: torch.tensor [batch_len, seq_len, vocab_len] - probability of next token for every sequence via log softmax (for loss function, which is nll_loss)
        """
        token_embeddings = self.token_embedding(input)
        batch_len, seq_len, embedding_dim = token_embeddings.size()
        data = self.positional_encoding(token_embeddings) # adds positional encodings to the token embeddings
        x = self.model(data)
        y = self.output(x.view(batch_len * seq_len, embedding_dim)).view(batch_len, seq_len, self.vocab_len)
        softmaxed = F.log_softmax(y, dim=2)
        return softmaxed
