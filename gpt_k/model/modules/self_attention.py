import torch
import torch.nn as nn
from .rotary_embedding import RotaryEmbedding, apply_rotary_pos_emb
import math

class SelfAttention(nn.Module):
    def __init__(self, args, use_cache=False, device=None):
        super().__init__()
        self.embedding_dim = args["embedding_dim"]
        self.use_cache = use_cache
        self.num_attention_heads = args["num_attention_heads"]
        self.embedding_dim_per_attention_head = args["embedding_dim"] // args["num_attention_heads"]
        self.rotary_ndims = int(self.embedding_dim_per_attention_head * args["rotary_pct"])
        self.rotary_emb = RotaryEmbedding(
            self.rotary_ndims,
            base=args["rotary_emb_base"],
            device=device,
        )
        self.query_key_value = nn.Linear(
            args["embedding_dim"],
            3 * args["embedding_dim"],
            device=device,
        )
        self.norm_factor = math.sqrt(self.embedding_dim_per_attention_head)
        self.dense = nn.Linear(
            args["embedding_dim"],
            args["embedding_dim"],
            device=device,
        )
        self.divide_qkv = args["divide_qkv"]

    def forward(self, hidden_states, attention_mask, layer_past=None):
        has_layer_past = layer_past is not None and layer_past.numel() > 0

        # Compute QKV
        # Attention heads [sq, b, h] --> [sq, b, (np * 3 * hn)]
        qkv = self.query_key_value(hidden_states)

        # [sq, b, (np * 3 * hn)] --> [sq, b, np, 3 * hn]
        new_qkv_shape = qkv.size()[:-1] + (
            self.num_attention_heads,
            3 * self.embedding_dim_per_attention_head,
        )
        qkv = qkv.view(*new_qkv_shape)

        # [sq, b, np, 3 * hn] --> 3 [sq, b, np, hn]
        query_layer = qkv[..., :self.embedding_dim_per_attention_head]
        key_layer = qkv[..., self.embedding_dim_per_attention_head: 2 * self.embedding_dim_per_attention_head]
        value_layer = qkv[..., 2 * self.embedding_dim_per_attention_head:]

        # Compute rotary embeddings
        query_rot, query_pass = (
            query_layer[..., : self.rotary_ndims],
            query_layer[..., self.rotary_ndims:],
        )
        key_rot, key_pass = (
            key_layer[..., : self.rotary_ndims],
            key_layer[..., self.rotary_ndims:],
        )
        seq_len = key_layer.shape[0]
        offset = 0
        if has_layer_past:
            offset = layer_past[0].shape[0]
            seq_len += offset
        cos, sin = self.rotary_emb(value_layer, seq_len=seq_len)
        query_layer, key_layer = apply_rotary_pos_emb(
            query_rot, key_rot, cos, sin, offset=offset,
        )
        query_layer = torch.cat((query_layer, query_pass), dim=-1)
        key_layer = torch.cat((key_layer, key_pass), dim=-1)

        # Cache QKV values
        if has_layer_past:
            past_key, past_value = layer_past
            key_layer = torch.cat((past_key.type_as(key_layer), key_layer), dim=0)
            value_layer = torch.cat((past_value.type_as(value_layer), value_layer), dim=0)
        if self.use_cache:
            kv_cache = torch.stack((key_layer, value_layer))
        else:
            kv_cache = None

        # Compute attention
        # noinspection PyTypeChecker
        context_layer = self.attention(
            query_layer, key_layer, value_layer, attention_mask
        )

        # Reshape outputs
        # [b, np, sq, hn] --> [sq, b, np, hn]
        context_layer = context_layer.permute(2, 0, 1, 3).contiguous()

        # [sq, b, np, hn] --> [sq, b, hp]
        new_context_layer_shape = context_layer.size()[:-2] + (
            self.embedding_dim,
        )
        context_layer = context_layer.view(*new_context_layer_shape)

        # =================
        # Output. [sq, b, h]
        # =================
        output = self.dense(context_layer)

        return output, kv_cache

    def attention(self, query_layer, key_layer, value_layer, attention_mask):
        # ===================================
        # Raw attention scores. [b, np, s, s]
        # ===================================

        # [b, np, sq, sk]
        output_size = (
            query_layer.size(1),
            query_layer.size(2),
            query_layer.size(0),
            key_layer.size(0),
        )

        # [sq, b, np, hn] -> [sq, b * np, hn]
        query_layer = query_layer.view(
            output_size[2], output_size[0] * output_size[1], -1
        )
        key_layer = key_layer.view(output_size[3], output_size[0] * output_size[1], -1)

        # preallocating result tensor: [b * np, sq, sk]
        matmul_result = torch.empty(
            output_size[0] * output_size[1],
            output_size[2],
            output_size[3],
            dtype=query_layer.dtype,
            device=query_layer.device,
        )

        # Raw attention scores. [b * np, sq, sk]
        matmul_result = torch.baddbmm(
            matmul_result,
            query_layer.transpose(0, 1),  # [b * np, sq, hn]
            key_layer.transpose(0, 1).transpose(1, 2),  # [b * np, hn, sk]
            beta=0.0,
            alpha=(1.0 / self.norm_factor),
        )

        # change view to [b, np, sq, sk]
        attention_scores = matmul_result.view(*output_size)

        # ==================================================
        # Update attention mask for inference. [b, np, sq, sk]
        # ==================================================

        # ===========================
        # Attention probs and dropout
        # ===========================

        # attention scores and attention mask [b, np, sq, sk]
        masked_scores = attention_mask_func(attention_scores, attention_mask) \
            if attention_mask is not None else attention_scores
        attention_probs = torch.nn.Softmax(dim=-1)(masked_scores)

        #         # This is actually dropping out entire tokens to attend to, which might
        #         # seem a bit unusual, but is taken from the original Transformer paper.
        #         attention_probs = self.attention_dropout(attention_probs)

        # =========================
        # Context layer. [sq, b, hp]
        # =========================

        # value_layer -> context layer.
        # [sk, b, np, hn] --> [b, np, sq, hn]

        # context layer shape: [b, np, sq, hn]
        output_size = (
            value_layer.size(1),
            value_layer.size(2),
            query_layer.size(0),
            value_layer.size(3),
        )

        # change view [sk, b * np, hn]
        value_layer = value_layer.view(
            value_layer.size(0), output_size[0] * output_size[1], -1
        )

        # change view [b * np, sq, sk]
        attention_probs = attention_probs.view(
            output_size[0] * output_size[1], output_size[2], -1
        )

        # matmul: [b * np, sq, hn]
        context_layer = torch.bmm(attention_probs, value_layer.transpose(0, 1))

        # change view [b, np, sq, hn]
        context_layer = context_layer.view(*output_size)
        return context_layer

def attention_mask_func(attention_scores, ltor_mask):
    """Assign -10000.0 to False cells in ltor_mask"""
    attention_scores.masked_fill_(~ltor_mask, -10000.0)
    return attention_scores

class JSelfAttention(nn.Module):
    def __init__(self, args, use_cache=False, device=None):
        super().__init__()
        self.embedding_dim = args["embedding_dim"]
        self.use_cache = use_cache
        self.num_attention_heads = args["num_attention_heads"]
        self.embedding_dim_per_attention_head = args["embedding_dim"] // args["num_attention_heads"]
        self.rotary_ndims = int(self.embedding_dim_per_attention_head * args["rotary_pct"])
        self.rotary_emb = RotaryEmbedding(
            self.rotary_ndims,
            base=args["rotary_emb_base"],
            device=device,
        )
        self.norm_factor = math.sqrt(self.embedding_dim_per_attention_head)
        self.dense = nn.Linear(
            args["embedding_dim"],
            args["embedding_dim"],
            device=device,
            bias=False # False on GPT-J
        )
        self.divide_qkv = args["divide_qkv"]
        if self.divide_qkv:
            self.query = nn.Linear(
                args["embedding_dim"],
                args["embedding_dim"],
                bias=False
            )
            self.key = nn.Linear(
                args["embedding_dim"],
                args["embedding_dim"],
                bias=False
            )
            self.value = nn.Linear(
                args["embedding_dim"],
                args["embedding_dim"],
                bias=False
            )
        else:
            self.query_key_value = nn.Linear(
                args["embedding_dim"],
                3 * args["embedding_dim"],
                device=device,
            )

    def forward(self, hidden_states, attention_mask, layer_past=None):
        has_layer_past = layer_past is not None and layer_past.numel() > 0

        if self.divide_qkv:
            q = self.query(hidden_states)
            k = self.key(hidden_states)
            v = self.value(hidden_states)
            qkv = torch.cat([q, k, v], dim=-1)
        else:
            # Compute QKV
            # Attention heads [sq, b, h] --> [sq, b, (np * 3 * hn)]
            qkv = self.query_key_value(hidden_states)

        # [sq, b, (np * 3 * hn)] --> [sq, b, np, 3 * hn]
        new_qkv_shape = qkv.size()[:-1] + (
            self.num_attention_heads,
            3 * self.embedding_dim_per_attention_head,
        )
        qkv = qkv.view(*new_qkv_shape)

        # [sq, b, np, 3 * hn] --> 3 [sq, b, np, hn]
        query_layer = qkv[..., :self.embedding_dim_per_attention_head]
        key_layer = qkv[..., self.embedding_dim_per_attention_head: 2 * self.embedding_dim_per_attention_head]
        value_layer = qkv[..., 2 * self.embedding_dim_per_attention_head:]

        # Compute rotary embeddings
        query_rot, query_pass = (
            query_layer[..., : self.rotary_ndims],
            query_layer[..., self.rotary_ndims:],
        )
        key_rot, key_pass = (
            key_layer[..., : self.rotary_ndims],
            key_layer[..., self.rotary_ndims:],
        )
        seq_len = key_layer.shape[0]
        offset = 0
        if has_layer_past:
            offset = layer_past[0].shape[0]
            seq_len += offset
        cos, sin = self.rotary_emb(value_layer, seq_len=seq_len)
        query_layer, key_layer = apply_rotary_pos_emb(
            query_rot, key_rot, cos, sin, offset=offset,
        )
        query_layer = torch.cat((query_layer, query_pass), dim=-1)
        key_layer = torch.cat((key_layer, key_pass), dim=-1)

        # Cache QKV values
        if has_layer_past:
            past_key, past_value = layer_past
            key_layer = torch.cat((past_key.type_as(key_layer), key_layer), dim=0)
            value_layer = torch.cat((past_value.type_as(value_layer), value_layer), dim=0)
        if self.use_cache:
            kv_cache = torch.stack((key_layer, value_layer))
        else:
            kv_cache = None

        # Compute attention
        # noinspection PyTypeChecker
        context_layer = self.attention(
            query_layer, key_layer, value_layer, attention_mask
        )

        # Reshape outputs
        # [b, np, sq, hn] --> [sq, b, np, hn]
        context_layer = context_layer.permute(2, 0, 1, 3).contiguous()

        # [sq, b, np, hn] --> [sq, b, hp]
        new_context_layer_shape = context_layer.size()[:-2] + (
            self.embedding_dim,
        )
        context_layer = context_layer.view(*new_context_layer_shape)

        # =================
        # Output. [sq, b, h]
        # =================
        output = self.dense(context_layer)
        
        if self.use_cache:
            return output, kv_cache
        else:
            return output

    def attention(self, query_layer, key_layer, value_layer, attention_mask):
        # ===================================
        # Raw attention scores. [b, np, s, s]
        # ===================================

        # [b, np, sq, sk]
        output_size = (
            query_layer.size(1),
            query_layer.size(2),
            query_layer.size(0),
            key_layer.size(0),
        )

        # [sq, b, np, hn] -> [sq, b * np, hn]
        query_layer = query_layer.view(
            output_size[2], output_size[0] * output_size[1], -1
        )
        key_layer = key_layer.view(output_size[3], output_size[0] * output_size[1], -1)

        # preallocating result tensor: [b * np, sq, sk]
        matmul_result = torch.empty(
            output_size[0] * output_size[1],
            output_size[2],
            output_size[3],
            dtype=query_layer.dtype,
            device=query_layer.device,
        )

        # Raw attention scores. [b * np, sq, sk]
        matmul_result = torch.baddbmm(
            matmul_result,
            query_layer.transpose(0, 1),  # [b * np, sq, hn]
            key_layer.transpose(0, 1).transpose(1, 2),  # [b * np, hn, sk]
            beta=0.0,
            alpha=(1.0 / self.norm_factor),
        )

        # change view to [b, np, sq, sk]
        attention_scores = matmul_result.view(*output_size)

        # ==================================================
        # Update attention mask for inference. [b, np, sq, sk]
        # ==================================================

        # ===========================
        # Attention probs and dropout
        # ===========================

        # attention scores and attention mask [b, np, sq, sk]
        masked_scores = attention_mask_func(attention_scores, attention_mask) \
            if attention_mask is not None else attention_scores
        attention_probs = torch.nn.Softmax(dim=-1)(masked_scores)

        #         # This is actually dropping out entire tokens to attend to, which might
        #         # seem a bit unusual, but is taken from the original Transformer paper.
        #         attention_probs = self.attention_dropout(attention_probs)

        # =========================
        # Context layer. [sq, b, hp]
        # =========================

        # value_layer -> context layer.
        # [sk, b, np, hn] --> [b, np, sq, hn]

        # context layer shape: [b, np, sq, hn]
        output_size = (
            value_layer.size(1),
            value_layer.size(2),
            query_layer.size(0),
            value_layer.size(3),
        )

        # change view [sk, b * np, hn]
        value_layer = value_layer.view(
            value_layer.size(0), output_size[0] * output_size[1], -1
        )

        # change view [b * np, sq, sk]
        attention_probs = attention_probs.view(
            output_size[0] * output_size[1], output_size[2], -1
        )

        # matmul: [b * np, sq, hn]
        context_layer = torch.bmm(attention_probs, value_layer.transpose(0, 1))

        # change view [b, np, sq, hn]
        context_layer = context_layer.view(*output_size)
        return context_layer

def attention_mask_func(attention_scores, ltor_mask):
    """Assign -10000.0 to False cells in ltor_mask"""
    attention_scores.masked_fill_(~ltor_mask, -10000.0)
    return attention_scores