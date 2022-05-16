import torch.nn as nn
from .self_attention import SelfAttention
from .mlp import MLP

class TransformerBlock(nn.Module):
    def __init__(self, args, device=None):
        super().__init__()
        self.layernorm_1 = nn.LayerNorm(
            args["embedding_dim"],
            eps=args["layernorm_eps"],
            device=device,
        )
        self.use_pa_ln = args["use_pa_ln"]
        if self.use_pa_ln:
            self.layernorm_2 = nn.LayerNorm(
                args["embedding_dim"],
                eps=args["layernorm_eps"],
                device=device,
            )
        self.attention = SelfAttention(args, device=device)
        self.mlp = MLP(args)

    def forward(self, X, attention_mask, layer_past=None):
        """
        Args:
            X: torch.Tensor [batch_len, seq_len, embedding_dim] = embedded input
        """
        residual = X
        hidden_states = self.layernorm_1(X)
        attention_output = self.attention(
            hidden_states,
            attention_mask,
        )
        hidden_states = attention_output + residual
        residual = hidden_states
        if self.use_pa_ln:
            hidden_states = self.layernorm_2(hidden_states)
        mlp_output = self.mlp(hidden_states)
        output = mlp_output + residual
        return output