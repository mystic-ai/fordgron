import torch
from torch import nn
import torch.nn.functional as F


class RotaryEmbedding(nn.Module):
    """
    Rotary embedding.

    Attributes:
        register_buffer: i don't know
        seq_len_cached: int - historic sequence length, stored
        cos_cached: float - historic cos, stored
        sin_cached: float - historic sin, stored
    """
    def __init__(self, num_rotary_dims=None, device=None, base=10000):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, num_rotary_dims, 2).float() / num_rotary_dims))
        self.register_buffer("inv_freq", inv_freq)
        self.highest_seq_len_cached = None
        self.cos_cached = None
        self.sin_cached = None
        self.num_rotary_dims = num_rotary_dims

    def _rotate_half(x):
      x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
      return torch.cat(
          (-x2, x1), dim=x1.ndim - 1
      )

    def forward(self, queries, keys, offset=0, seq_len=None):
        """
        """
        if exists(self.num_rotary_dims):
            # partial rotary
            queries_for_rotation = queries_layer[..., :self.num_rotary_dims] # all up to self.num_rotary_dims
            queries_skipped = queries_layer[..., self.num_rotary_dims:] # all after self.num_rotary_dims
            keys_for_rotation = keys_layer[..., :self.num_rotary_dims]
            keys_skipped = keys_layer[..., self.num_rotary_dims:]
        else:
            # full rotary
            queries_for_rotation, keys_for_rotation = queries, keys

        offset = 0
        # work on layer_past logic later
        """  if exists(layer_past) and layer_past.numel() > 0:
            offset = layer_past[0].shape[0]
            seq_len += offset """

        if seq_len < self.highest_seq_len_cached:
            self.highest_seq_len_cached = seq_len
            t = torch.arange(self.highest_seq_len_cached, device=X.device, dtype=self.inv_freq.dtype)
            frequencies = torch.einsum("i,j->ij", t, self.inv_freq)
            emb = torch.cat((frequencies, frequencies), dim=-1).to(X.device)
            # [sx, 1 (b * np), hn]
            self.cos_cached = emb.cos()[:, None, None, :]
            self.sin_cached = emb.sin()[:, None, None, :]
            # not sure what's going on in the above
        cos = self.cos_cached[offset:seq_len + offset, ...]
        sin = self.sin_cached[offset:seq_len + offset, ...]
        rotated_queries = (queries * cos) + (_rotate_half(queries) * sin)
        rotated_keys = (keys * cos) + (_rotate_half(keys) * sin)
        if exists(self.num_rotary_dims):
            rotated_queries = torch.cat((rotated_queries, queries_skipped), dim=-1)
            rotated_keys = torch.cat((rotated_keys, keys_skipped), dim=-1)
        return rotated_queries, rotated_keys