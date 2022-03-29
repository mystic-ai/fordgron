import torch
from torch import nn
import torch.nn.functional as F
from rich.progress import Progress
from .modules import PositionalEncoding, TransformerBlock
import time

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
        num_attention_heads,
        mask,
        dropout,
        forward_expansion,
        depth,
        seq_len,
        vocab_len,
        positional_encoding_implementation = "positional_encoding",
        rotary_pct = 0,
        layernorm_epsilon = 1e-5,
        device = None
    ):
        super().__init__()
        with Progress() as progress:
            task1 = progress.add_task("building transformer", total=depth + 5 ) # there are five steps beyond stacking transformer layers
            self.vocab_len = vocab_len
            self.token_embedding = nn.Embedding(vocab_len, embedding_dim)
            progress.update(task1, advance=1)
            self.positional_encoding = PositionalEncoding(dropout, seq_len, embedding_dim)
            self.positional_encoding_implementation = positional_encoding_implementation
            progress.update(task1, advance=2)
            modules = []
            for _ in range(depth):
                modules.append(                
                    TransformerBlock(
                        embedding_dim,
                        num_attention_heads,
                        mask,
                        dropout,
                        forward_expansion,
                        positional_encoding_implementation,
                        rotary_pct,
                        layernorm_epsilon,
                        device
                    )
                )
                progress.update(task1, advance=1)
            self.transformer_stack = nn.Sequential(*modules)
            progress.update(task1, advance=1)
            self.final_layer_norm = nn.LayerNorm(
                embedding_dim,
                eps=layernorm_epsilon
            )
            progress.update(task1, advance=1)
            self.output_logits = nn.Linear(embedding_dim, self.vocab_len, bias=False) # given predicted token embeddings, what's the next word?
            progress.update(task1, advance=1)

    def forward(self, input, attention_mask=False):
        """
        Args:
            input: torch.tensor [batch_len, seq_len] - matrix of token sequences
        Returns:
            softmaxed: torch.tensor [batch_len, seq_len, vocab_len] - probability of next token for every sequence via log softmax (for loss function, which is nll_loss)
        """
        X = self.token_embedding(input) # [batch_len, seq_len, embedding_dim]
        if (self.positional_encoding_implementation == "positional_encoding"):
            data = self.positional_encoding(data) # adds positional encodings to the token embeddings
        X = self.swap_batch_len_and_seq_len(X)
        X = self.transformer_stack(X)
        X = self.swap_batch_len_and_seq_len(X)
        X = self.final_layer_norm(X) # [batch_len, seq_len, embedding_dim]
        logits = self.output_logits(X) # [batch_len, seq_len, vocab_len]
        return logits

    @classmethod
    def swap_batch_len_and_seq_len(cls, X):
        return X.transpose(0, 1).contiguous()