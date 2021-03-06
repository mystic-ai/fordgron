import torch.nn as nn
from .self_attention import JSelfAttention
from .mlp import MLP

class TransformerBlock(nn.Module):
    def __init__(self, args, use_cache, device=None):
        super().__init__()
        self.use_cache = use_cache
        self.input_layernorm = nn.LayerNorm(
            args["embedding_dim"],
            eps=args["layernorm_eps"],
            device=device,
        )
        self.use_pa_ln = args["use_pa_ln"]
        if self.use_pa_ln:
            self.post_attention_layernorm = nn.LayerNorm(
                args["embedding_dim"],
                eps=args["layernorm_eps"],
                device=device,
            )
        self.attention = JSelfAttention(args, self.use_cache, device=device)
        self.mlp = MLP(args)

    def forward(self, x, attention_mask, layer_past=None):
        """
        Args:
            x: torch.Tensor [batch_len, seq_len, embedding_dim] = embedded input
        """
        residual = x
        x = self.input_layernorm(x)
        if self.use_cache:
            attention_output, kv_cache = self.attention(
                x,
                attention_mask,
                layer_past=layer_past,
            )
        else:
            attention_output = self.attention(
                x,
                attention_mask,
                layer_past=layer_past,
            )
        """         if self.use_pa_ln:
            x = self.post_attention_layernorm(x) """
        print("attention output")
        print(attention_output)
        mlp_output = self.mlp(hidden_states=x)
        print("mlp output")
        print(mlp_output)
        print("residual")
        print(residual)
        output = residual + mlp_output + attention_output
        print("grouped")
        print(output)
        if self.use_cache:
            return output, kv_cache
        else:
            return output