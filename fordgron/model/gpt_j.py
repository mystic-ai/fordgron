""" PyTorch GPT-J model."""

from typing import Optional, Tuple, Union

import torch
from torch import nn, Tensor
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
import math


class NewGELUActivation(nn.Module):
    """
    Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT). Also see
    the Gaussian Error Linear Units paper: https://arxiv.org/abs/1606.08415
    """

    def forward(self, input: Tensor) -> Tensor:
        return (
            0.5
            * input
            * (
                1.0
                + torch.tanh(
                    math.sqrt(2.0 / math.pi)
                    * (input + 0.044715 * torch.pow(input, 3.0))
                )
            )
        )


def fixed_pos_embedding(x, seq_dim=1, seq_len=None):
    dim = x.shape[-1]
    if seq_len is None:
        seq_len = x.shape[seq_dim]
    inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2) / dim))
    sinusoid_inp = (
        torch.einsum("i , j -> i j", torch.arange(seq_len, dtype=torch.float), inv_freq)
        .to(x.device)
        .float()
    )
    return torch.sin(sinusoid_inp), torch.cos(sinusoid_inp)


def rotate_every_two(x):
    x1 = x[:, :, :, ::2]
    x2 = x[:, :, :, 1::2]
    x = torch.stack((-x2, x1), dim=-1)
    return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')


def duplicate_interleave(m):
    """
    A simple version of `torch.repeat_interleave` for duplicating a matrix while interleaving the copy.
    """
    dim0 = m.shape[0]
    m = m.view(-1, 1)  # flatten the matrix
    m = m.repeat(1, 2)  # repeat all elements into the 2nd dimension
    m = m.view(dim0, -1)  # reshape into a matrix, interleaving the copy
    return m


def apply_rotary_pos_emb(x, sincos, offset=0):
    sin, cos = map(
        lambda t: duplicate_interleave(t)[None, offset : x.shape[1] + offset, None, :],
        sincos,
    )
    # einsum notation for lambda t: repeat(t[offset:x.shape[1]+offset,:], "n d -> () n () (d j)", j=2)
    return (x * cos) + (rotate_every_two(x) * sin)


class GPTJAttention(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.seq_len = args["seq_len"]
        self.bias = None  # defined during forward pass
        self.masked_bias = None  # ditto

        self.embedding_dim = args["embedding_dim"]
        self.num_attention_heads = args["num_attention_heads"]
        self.embedding_dim_per_attention_head = (
            self.embedding_dim // self.num_attention_heads
        )
        if (
            self.embedding_dim_per_attention_head * self.num_attention_heads
            != self.embedding_dim
        ):
            raise ValueError(
                f"embedding_dim must be divisible by num_attention_heads (got `embedding_dim`: {self.embedding_dim} and"
                f" `num_attention_heads`: {self.num_attention_heads})."
            )
        self.scale_attn = torch.sqrt(
            torch.tensor(self.embedding_dim_per_attention_head, dtype=torch.float32)
        ).to(torch.get_default_dtype())

        self.k_proj = nn.Linear(self.embedding_dim, self.embedding_dim, bias=False)
        self.v_proj = nn.Linear(self.embedding_dim, self.embedding_dim, bias=False)
        self.q_proj = nn.Linear(self.embedding_dim, self.embedding_dim, bias=False)
        self.out_proj = nn.Linear(self.embedding_dim, self.embedding_dim, bias=False)
        self.rotary_dim = (
            64  # self.embedding_dim_per_attention_head * args["rotary_pct"]
        )

    def _split_heads(
        self, tensor, num_attention_heads, embedding_dim_per_attention_head, rotary
    ):
        """
        Splits hidden dim into embedding_dim_per_attention_head and num_attention_heads
        """
        new_shape = tensor.size()[:-1] + (
            num_attention_heads,
            embedding_dim_per_attention_head,
        )
        tensor = tensor.view(new_shape)
        if rotary:
            return tensor
        if len(tensor.shape) == 5:
            return tensor.permute(
                0, 1, 3, 2, 4
            )  # (batch, blocks, head, block_length, head_features)
        elif len(tensor.shape) == 4:
            return tensor.permute(
                0, 2, 1, 3
            )  # (batch, head, seq_length, head_features)
        else:
            raise ValueError(
                f"Input tensor rank should be one of [4, 5], but is: {len(tensor.shape)}"
            )

    def _merge_heads(
        self, tensor, num_attention_heads, embedding_dim_per_attention_head
    ):
        """
        Merges embedding_dim_per_attention_head dim and num_attn_heads dim into hidden dim
        """
        if len(tensor.shape) == 5:
            tensor = tensor.permute(0, 1, 3, 2, 4).contiguous()
        elif len(tensor.shape) == 4:
            tensor = tensor.permute(0, 2, 1, 3).contiguous()
        else:
            raise ValueError(
                f"Input tensor rank should be one of [4, 5], but is: {len(tensor.shape)}"
            )
        new_shape = tensor.size()[:-2] + (
            num_attention_heads * embedding_dim_per_attention_head,
        )
        return tensor.view(new_shape)

    def _attn(
        self,
        query,
        key,
        value,
        attention_mask=None,
        head_mask=None,
    ):

        # compute causal mask from causal mask buffer
        query_length, key_length = query.size(-2), key.size(-2)
        causal_mask = self.bias[
            :, :, key_length - query_length : key_length, :key_length
        ].to(torch.bool)

        # Keep the attention weights computation in fp32 to avoid overflow issues
        query = query.to(torch.float32)
        key = key.to(torch.float32)

        attn_weights = torch.matmul(query, key.transpose(-1, -2))
        attn_weights = torch.where(
            causal_mask, attn_weights, self.masked_bias.to(attn_weights.dtype)
        )

        attn_weights = attn_weights / self.scale_attn

        if attention_mask is not None:
            # Apply the attention mask
            attn_weights = attn_weights + attention_mask

        attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        attn_weights = attn_weights.to(value.dtype)

        # Mask heads if we want to
        if head_mask is not None:
            attn_weights = attn_weights * head_mask

        attn_output = torch.matmul(attn_weights, value)

        return attn_output, attn_weights

    def forward(
        self,
        hidden_states: Optional[torch.FloatTensor],
        attention_mask: Optional[torch.FloatTensor] = None,
        layer_past: Optional[Tuple[torch.Tensor]] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
    ) -> Union[
        Tuple[torch.Tensor, Tuple[torch.Tensor]],
        Optional[Tuple[torch.Tensor, Tuple[torch.Tensor], Tuple[torch.Tensor, ...]]],
    ]:

        if self.bias == None:
            self.bias = torch.tril(
                torch.ones(
                    (self.seq_len, self.seq_len),
                    dtype=torch.uint8,
                    device=hidden_states.device,
                )
            ).view(1, 1, self.seq_len, self.seq_len)
        if self.masked_bias == None:
            self.masked_bias = torch.tensor(-1e9, device=hidden_states.device)

        # layer past needs to be a tuple of (this layer's previous key, this layer's previous value)

        query = self.q_proj(hidden_states)
        key = self.k_proj(hidden_states)
        value = self.v_proj(hidden_states)

        query = self._split_heads(
            query, self.num_attention_heads, self.embedding_dim_per_attention_head, True
        )
        key = self._split_heads(
            key, self.num_attention_heads, self.embedding_dim_per_attention_head, True
        )
        value = self._split_heads(
            value,
            self.num_attention_heads,
            self.embedding_dim_per_attention_head,
            False,
        )

        seq_len = key.shape[1]
        offset = 0

        if layer_past is not None:
            offset = layer_past[0].shape[-2]
            seq_len += offset

        if self.rotary_dim is not None:
            k_rot = key[:, :, :, : self.rotary_dim]
            k_pass = key[:, :, :, self.rotary_dim :]

            q_rot = query[:, :, :, : self.rotary_dim]
            q_pass = query[:, :, :, self.rotary_dim :]

            sincos = fixed_pos_embedding(k_rot, 1, seq_len=seq_len)
            k_rot = apply_rotary_pos_emb(k_rot, sincos, offset=offset)
            q_rot = apply_rotary_pos_emb(q_rot, sincos, offset=offset)

            key = torch.cat([k_rot, k_pass], dim=-1)
            query = torch.cat([q_rot, q_pass], dim=-1)
        else:
            sincos = fixed_pos_embedding(key, 1, seq_len=seq_len)
            key = apply_rotary_pos_emb(key, sincos, offset=offset)
            query = apply_rotary_pos_emb(query, sincos, offset=offset)

        key = key.permute(0, 2, 1, 3)
        query = query.permute(0, 2, 1, 3)

        if layer_past is not None:
            past_key = layer_past[0]
            past_value = layer_past[1]
            key = torch.cat((past_key, key), dim=-2)
            value = torch.cat((past_value, value), dim=-2)

        if use_cache is True:
            present = (key, value)
        else:
            present = None

        # compute self-attention: V x Softmax(QK^T)
        attn_output, attn_weights = self._attn(
            query, key, value, attention_mask, head_mask
        )

        attn_output = self._merge_heads(
            attn_output, self.num_attention_heads, self.embedding_dim_per_attention_head
        )
        attn_output = self.out_proj(attn_output)

        outputs = (attn_output, present)
        if output_attentions:
            outputs += (attn_weights,)

        return outputs  # a, present, (attentions)


class GPTJMLP(nn.Module):
    def __init__(
        self, intermediate_size, args, device
    ):  # in MLP: intermediate_size= 4 * embedding_dim
        super().__init__()
        embedding_dim = args["embedding_dim"]

        self.fc_in = nn.Linear(embedding_dim, intermediate_size, device=device)
        self.fc_out = nn.Linear(intermediate_size, embedding_dim, device=device)

        self.act = NewGELUActivation()

    def forward(self, hidden_states: Optional[torch.FloatTensor]) -> torch.FloatTensor:
        hidden_states = self.fc_in(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.fc_out(hidden_states)
        return hidden_states


class GPTJBlock(nn.Module):
    def __init__(self, args, device):
        super().__init__()
        inner_dim = 4 * args["embedding_dim"]
        self.ln_1 = nn.LayerNorm(
            args["embedding_dim"], eps=args["layernorm_eps"], device=device
        )
        self.attn = GPTJAttention(args)
        self.mlp = GPTJMLP(inner_dim, args, device=device)

    def forward(
        self,
        hidden_states: Optional[torch.FloatTensor],
        layer_past: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
    ) -> Union[
        Tuple[torch.Tensor],
        Optional[Tuple[torch.Tensor, Tuple[torch.FloatTensor, ...]]],
    ]:

        # layer_past needs to be a tuple of (this block's previous generation key, this block's previous generation value)

        residual = hidden_states
        hidden_states = self.ln_1(hidden_states)
        attn_outputs = self.attn(
            hidden_states,
            layer_past=layer_past,
            attention_mask=attention_mask,
            head_mask=head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
        )
        attn_output = attn_outputs[0]  # output_attn: a, present, (attentions)
        outputs = attn_outputs[1:]
        feed_forward_hidden_states = self.mlp(hidden_states)
        hidden_states = residual + feed_forward_hidden_states + attn_output

        if use_cache:
            outputs = (hidden_states,) + outputs
        else:
            outputs = (hidden_states,) + outputs[1:]
        return outputs  # hidden_states, present, (attentions)


class GPTJModel(nn.Module):
    def __init__(self, args, device):
        super().__init__()

        self.embedding_dim = args["embedding_dim"]
        self.vocab_len = args["vocab_len"]
        self.token_embedding = nn.Embedding(
            args["vocab_len"], self.embedding_dim, device=device
        )
        self.h = nn.ModuleList(
            [GPTJBlock(args, device=device) for _ in range(args["depth"])]
        )
        self.ln_f = nn.LayerNorm(
            self.embedding_dim, eps=args["layernorm_eps"], device=device
        )
        self.depth = args["depth"]

    def get_input_embeddings(self):
        return self.token_embedding

    def set_input_embeddings(self, new_embeddings):
        self.token_embedding = new_embeddings

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Tensor:

        output_attentions = False
        output_hidden_states = False

        return_dict = False

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time"
            )
        elif input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
            batch_size = input_ids.shape[0]
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
            batch_size = inputs_embeds.shape[0]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if token_type_ids is not None:
            token_type_ids = token_type_ids.view(-1, input_shape[-1])

        if position_ids is not None:
            position_ids = position_ids.view(-1, input_shape[-1])

        if past_key_values is None:
            past_length = 0
            past_key_values = tuple([None] * len(self.h))
        else:
            past_length = past_key_values[0][0].size(-2)

        if position_ids is None:
            position_ids = torch.arange(
                past_length,
                input_shape[-1] + past_length,
                dtype=torch.long,
                device=device,
            )
            position_ids = position_ids.unsqueeze(0).view(-1, input_shape[-1])

        # Attention mask.
        if attention_mask is not None:
            if batch_size <= 0:
                raise ValueError("batch_size has to be defined and > 0")
            attention_mask = attention_mask.view(batch_size, -1)
            # We create a 3D attention mask from a 2D tensor mask.
            # Sizes are [batch_size, 1, 1, to_seq_length]
            # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
            # this attention mask is more simple than the triangular masking of causal attention
            # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
            attention_mask = attention_mask[:, None, None, :]

            # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
            # masked positions, this operation will create a tensor which is 0.0 for
            # positions we want to attend and -10000.0 for masked positions.
            # Since we are adding it to the raw scores before the softmax, this is
            # effectively the same as removing these entirely.
            attention_mask = attention_mask.to(dtype=self.dtype)  # fp16 compatibility
            attention_mask = (1.0 - attention_mask) * -10000.0

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x num_attention_heads x N x N
        # head_mask has shape n_layer x batch x num_attention_heads x N x N
        # head_mask = self.get_head_mask(head_mask, self.depth)

        if inputs_embeds is None:
            inputs_embeds = self.token_embedding(input_ids)

        hidden_states = inputs_embeds

        if token_type_ids is not None:
            token_type_embeds = self.token_embedding(token_type_ids)
            hidden_states = hidden_states + token_type_embeds

        output_shape = input_shape + (hidden_states.size(-1),)

        presents = () if use_cache else None
        all_self_attentions = () if output_attentions else None
        all_hidden_states = () if output_hidden_states else None
        for i, (block, layer_past) in enumerate(zip(self.h, past_key_values)):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            outputs = block(
                hidden_states,
                layer_past=layer_past,
                attention_mask=attention_mask,
                head_mask=head_mask,
                use_cache=use_cache,
                output_attentions=output_attentions,
            )

            hidden_states = outputs[0]
            if use_cache is True:
                presents = presents + (outputs[1],)

            if output_attentions:
                all_self_attentions = all_self_attentions + (
                    outputs[2 if use_cache else 1],
                )

        hidden_states = self.ln_f(hidden_states)
        hidden_states = hidden_states.view(output_shape)
        # Add last hidden state
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        return hidden_states, presents


class GPTJForCausalLM(nn.Module):
    _keys_to_ignore_on_load_missing = [
        r"h\.\d+\.attn\.masked_bias",
        r"h\.\d+\.attn\.bias",
    ]

    def __init__(self, args, device=None):
        super().__init__()
        self.transformer = GPTJModel(args, device=device)
        self.lm_head = nn.Linear(
            args["embedding_dim"], args["vocab_len"], device=device
        )

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Tensor:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for language modeling. Note that the labels **are shifted** inside the model, i.e. you can set
            `labels = input_ids` Indices are selected in `[-100, 0, ..., args.vocab_len]` All labels set to `-100`
            are ignored (masked), the loss is only computed for labels in `[0, ..., args.vocab_len]`
        """

        hidden_states, presents = self.transformer(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # make sure sampling in fp16 works correctly and
        # compute loss in fp32 to match with mesh-tf version
        # https://github.com/EleutherAI/gpt-neo/blob/89ce74164da2fb16179106f54e2269b5da8db333/models/gpt2/gpt2.py#L179
        lm_logits = self.lm_head(hidden_states).to(torch.float32)

        return lm_logits, presents
