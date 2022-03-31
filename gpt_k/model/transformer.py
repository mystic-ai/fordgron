import torch
from torch import nn
import torch.nn.functional as F
from rich.progress import Progress
from .modules import PositionalEncoding, TransformerBlock
import time

class TransformerStack(nn.Sequential):
    def forward(self, input, **kwargs):
        for module in self:
            input = module(input, **kwargs)
        return input

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
        eod_id=None,
        batch_len_and_seq_len_swapped=False,
        use_cache=False,
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
            modules = nn.ModuleList([])
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
                        use_cache=use_cache,
                        device=device
                    )
                )
                progress.update(task1, advance=1)
            self.transformer_stack = TransformerStack(*modules)
            progress.update(task1, advance=1)
            self.final_layer_norm = nn.LayerNorm(
                embedding_dim,
                eps=layernorm_epsilon
            )
            progress.update(task1, advance=1)
            self.output_logits = nn.Linear(embedding_dim, self.vocab_len, bias=False) # given predicted token embeddings, what's the next word?
            progress.update(task1, advance=1)
            self.eod_id = eod_id
            self.depth = depth
            self.use_cache = use_cache
            self.batch_len_and_seq_len_swapped = batch_len_and_seq_len_swapped

    def forward(self, input, attention_mask=None, previous_layer=None):
        """
        Args:
            input: torch.tensor [batch_len, seq_len] - matrix of token sequences
        Returns:
            softmaxed: torch.tensor [batch_len, seq_len, vocab_len] - probability of next token for every sequence via log softmax (for loss function, which is nll_loss)
        """
        if previous_layer is None:
            print("previous_layer not defined in transformer.py, so now it's defined")
            previous_layer = [None] * self.depth
            print(previous_layer)
        kv_cache_yeah = None

        print("input size")
        print(input.size())
        print("^^^")

        """
        build token embedding
        in the model weights there is a mapping from every token to an `embedding_dim` magnitude vector
        the nn.Embedding module merely performs a lookup on every token in the input and replaces it with its embedding vector 
        input [batch_len, seq_len] -> X [batch_len, seq_len, embedding_dim]
        """
        X = self.token_embedding(input) # token embedding is the same every time, yet recomputed each forward pass, can we fix this?

        """
        if (self.positional_encoding_implementation == "positional_encoding"):
            data = self.positional_encoding(data) # adds positional encodings to the token embeddings
        """

        """
        transpose the first two dimensions of X if the model weights require it
        certain models (i.e. Megatron) apparently were trained that way because it is more efficient, somehow, even though it's a pain in the neck
        POSSIBLY: X [batch_len, seq_len, embedding_dim] -> X [seq_len, batch_len, embedding_dim]
        """
        if self.batch_len_and_seq_len_swapped:
            X = self.swap_batch_len_and_seq_len(X) # [input_seq_len, batch_len, embedding_dim]

        """
        the transformer sandwich
        run X through every layer in the transformer stack
        """
        if self.use_cache:
            for layer_index, transformer_layer in enumerate(self.transformer_stack):
                X = transformer_layer(X, attention_mask)

        """
        if we transposed the first two dimensions of X earlier, then we need to transpose them again, returning them to their normal state
        POSSIBLY: X [seq_len, batch_len, embedding_dim] -> X [batch_len, seq_len, embedding_dim]
        """
        if self.batch_len_and_seq_len_swapped:
            X = self.swap_batch_len_and_seq_len(X) # [batch_len, input_seq_len, embedding_dim]

        X = self.final_layer_norm(X) # [batch_len, input_seq_len, embedding_dim]
        logits = self.output_logits(X) # [batch_len, input_seq_len, vocab_len]
        return logits

    @classmethod
    def swap_batch_len_and_seq_len(cls, X):
        return X.transpose(0, 1).contiguous()