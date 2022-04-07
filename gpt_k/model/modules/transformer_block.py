import torch.nn as nn
from .self_attention import SelfAttention
from .mlp import MLP

class TransformerBlock(nn.Module):
    def __init__(self, args, use_cache, device=None):
        super().__init__()
        self.use_cache = use_cache
        self.input_layernorm = nn.LayerNorm(
            args.hidden_size,
            eps=args.layernorm_epsilon,
            device=device,
        )
        self.post_attention_layernorm = nn.LayerNorm(
            args.hidden_size,
            eps=args.layernorm_epsilon,
            device=device,
        )
        self.attention = SelfAttention(args, self.use_cache, device=device)
        self.mlp = MLP(args)

    def forward(self, x, attention_mask, layer_past=None):
        residual = x
        ln_output = self.input_layernorm(x)
        attention_output, kv_cache = self.attention(
            ln_output,
            attention_mask,
            layer_past=layer_past,
        )
        post_attn_ln = self.post_attention_layernorm(x)
        mlp_output = self.mlp(hidden_states=post_attn_ln)
        output = residual + mlp_output + attention_output
        return output, kv_cache