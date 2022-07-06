from typing import List
import torch
from torch import nn, BoolTensor, FloatTensor, LongTensor
from transformers import BartTokenizerFast
torch.set_grad_enabled(False)


class GLU(nn.Module):
    def __init__(self, count_in_out, count_middle):
        super().__init__()
        self.gelu = nn.GELU()
        self.ln0 = nn.LayerNorm(count_in_out)
        self.ln1 = nn.LayerNorm(count_middle)
        self.fc0 = nn.Linear(count_in_out, count_middle, bias=False)
        self.fc1 = nn.Linear(count_in_out, count_middle, bias=False)
        self.fc2 = nn.Linear(count_middle, count_in_out, bias=False)
    
    def forward(self, z: FloatTensor) -> FloatTensor:
        z = self.ln0.forward(z)
        w = self.fc0.forward(z)
        w = self.gelu.forward(w)
        v = self.fc1.forward(z)
        z = self.ln1.forward(w * v)
        z = self.fc2.forward(z)
        return z


class AttentionBase(nn.Module):
    def __init__(self, head_count: int, embedding_dim: int):
        super().__init__()
        self.head_count = head_count
        self.embedding_dim = embedding_dim

        self.k_proj = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.v_proj = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.q_proj = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.out_proj = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.one = torch.ones((1, 1))
        if torch.cuda.is_available(): self.one = self.one.cuda()
    
    def forward(
        self,
        keys: FloatTensor,
        values: FloatTensor,
        queries: FloatTensor,
        attention_mask: BoolTensor
    ) -> FloatTensor:
        keys = keys.reshape(keys.shape[:2] + (self.head_count, -1))
        values = values.reshape(values.shape[:2] + (self.head_count, -1))
        queries = queries.reshape(queries.shape[:2] + (self.head_count, -1))
        queries /= queries.shape[-1] ** 0.5

        attention_bias = torch.where(
            attention_mask,
            self.one * 0,
            self.one * (-torch.inf),
        )
        attention_weights: FloatTensor = torch.einsum(
            'bqhc,bkhc->bhqk',
            queries, 
            keys
        )
        attention_weights += attention_bias[:, None, None, :]
        attention_weights = torch.softmax(attention_weights, -1)
        attention_output: FloatTensor = torch.einsum(
            "bhqk,bkhc->bqhc",
            attention_weights, 
            values
        )
        shape = attention_output.shape[:2] + (self.embedding_dim,)
        attention_output = attention_output.reshape(shape)
        attention_output = self.out_proj.forward(attention_output)
        return attention_output


class EncoderSelfAttention(AttentionBase):
    def forward(
        self,
        encoder_state: FloatTensor,
        attention_mask: BoolTensor
    ) -> FloatTensor:
        keys = self.k_proj.forward(encoder_state)
        values = self.v_proj.forward(encoder_state)
        queries = self.q_proj.forward(encoder_state)
        return super().forward(keys, values, queries, attention_mask)


class EncoderLayer(nn.Module):
    def __init__(self, embedding_dim: int, head_count: int, glu_embedding_dim: int):
        super().__init__()
        self.pre_self_attn_layer_norm = nn.LayerNorm(embedding_dim)
        self.self_attn = EncoderSelfAttention(head_count, embedding_dim)
        self.self_attn_layer_norm = nn.LayerNorm(embedding_dim)
        self.glu = GLU(embedding_dim, glu_embedding_dim)
    
    def forward(
        self,
        encoder_state: FloatTensor,
        attention_mask: BoolTensor
    ) -> FloatTensor:
        residual = encoder_state
        encoder_state = self.pre_self_attn_layer_norm.forward(encoder_state)
        encoder_state = self.self_attn.forward(encoder_state, attention_mask)
        encoder_state = self.self_attn_layer_norm.forward(encoder_state)
        encoder_state = residual + encoder_state
        residual = encoder_state
        encoder_state = self.glu.forward(encoder_state)
        encoder_state = residual + encoder_state
        return encoder_state


class DalleBartEncoder(nn.Module):
    def __init__(
        self,
        depth: int,
        embedding_dim: int,
        num_attention_heads: int,
        text_vocab_len: int,
        text_token_count: int,
        glu_embedding_dim: int
    ):
        super().__init__()
        self.embed_tokens = nn.Embedding(text_vocab_len, embedding_dim)
        self.embed_positions = nn.Embedding(text_token_count, embedding_dim)
        self.layers: List[EncoderLayer] = nn.ModuleList([
            EncoderLayer(
                embedding_dim = embedding_dim,
                head_count = num_attention_heads,
                glu_embedding_dim = glu_embedding_dim
            ) 
            for _ in range(depth)
        ])
        self.layernorm_embedding = nn.LayerNorm(embedding_dim)
        self.final_ln = nn.LayerNorm(embedding_dim)
        self.token_indices = torch.arange(text_token_count).to(torch.long)
        if torch.cuda.is_available(): 
            self.token_indices = self.token_indices.cuda()

    def forward(self, text_tokens: LongTensor) -> FloatTensor:
        attention_mask = text_tokens.not_equal(1)
        pose_tokens = self.token_indices[None][[0] * text_tokens.shape[0]]
        encoder_state = (
            self.embed_tokens.forward(text_tokens) +
            self.embed_positions.forward(pose_tokens)
        )
        encoder_state = self.layernorm_embedding.forward(encoder_state)
        for layer in self.layers:
            encoder_state = layer.forward(encoder_state, attention_mask)
        encoder_state = self.final_ln.forward(encoder_state)
        return encoder_state


from math import inf
from typing import List, Tuple

class TextTokenizer:
    def __init__(self, vocab: dict, merges: List[str]):
        self.token_from_subword = vocab
        pairs = [tuple(pair.split()) for pair in merges]
        self.rank_from_pair = dict(zip(pairs, range(len(pairs))))

    def tokenize(self, text: str, is_verbose: bool = False) -> List[int]:
        sep_token = self.token_from_subword['</s>']
        cls_token = self.token_from_subword['<s>']
        unk_token = self.token_from_subword['<unk>']
        text = text.encode("ascii", errors="ignore").decode()
        tokens = [
            self.token_from_subword.get(subword, unk_token)
            for word in text.split(" ") if len(word) > 0
            for subword in self.get_byte_pair_encoding(word, is_verbose)
        ]
        return [cls_token] + tokens + [sep_token]

    def get_byte_pair_encoding(self, word: str, is_verbose: bool) -> List[str]:
        def get_pair_rank(pair: Tuple[str, str]) -> int:
            return self.rank_from_pair.get(pair, inf)

        subwords = [chr(ord(" ") + 256)] + list(word)
        while len(subwords) > 1:
            pairs = list(zip(subwords[:-1], subwords[1:]))
            pair_to_merge = min(pairs, key=get_pair_rank)
            if pair_to_merge not in self.rank_from_pair: break
            i = pairs.index(pair_to_merge)
            subwords = (
                (subwords[:i] if i > 0 else []) + 
                [subwords[i] + subwords[i + 1]] + 
                (subwords[i + 2:] if i + 2 < len(subwords) else [])
            )

        if is_verbose: print(subwords)
        return subwords

from typing import Tuple, List
import torch
from torch import LongTensor, nn, FloatTensor, BoolTensor
torch.set_grad_enabled(False)

IMAGE_TOKEN_COUNT = 256
BLANK_TOKEN = 6965


class DecoderCrossAttention(AttentionBase):
    def forward(
        self,
        decoder_state: FloatTensor,
        encoder_state: FloatTensor,
        attention_mask: BoolTensor
    ) -> FloatTensor:
        keys = self.k_proj.forward(encoder_state)
        values = self.v_proj.forward(encoder_state)
        queries = self.q_proj.forward(decoder_state)
        return super().forward(keys, values, queries, attention_mask)


class DecoderSelfAttention(AttentionBase):
    def __init__(self, head_count: int, embedding_dim: int):
        super().__init__(head_count, embedding_dim)
        token_indices = torch.arange(IMAGE_TOKEN_COUNT)
        if torch.cuda.is_available(): token_indices = token_indices.cuda()
        self.token_indices = token_indices

    def forward(
        self, 
        decoder_state: FloatTensor,
        attention_state: FloatTensor,
        token_index: LongTensor
    ) -> Tuple[FloatTensor, FloatTensor]:
        keys = self.k_proj.forward(decoder_state)
        values = self.v_proj.forward(decoder_state)
        queries = self.q_proj.forward(decoder_state)
        attn_mask = self.token_indices < token_index + 1
        attn_mask = attn_mask[None][[0] * decoder_state.shape[0]]
        attention_state[:, token_index] = torch.cat([keys, values])
        batch_count = decoder_state.shape[0]
        keys = attention_state[:batch_count]
        values = attention_state[batch_count:]
        decoder_state = super().forward(keys, values, queries, attn_mask)
        return decoder_state, attention_state


class DecoderLayer(nn.Module):
    def __init__(
        self, 
        head_count: int, 
        embedding_dim: int,
        glu_embedding_dim: int
    ):
        super().__init__()
        self.pre_self_attn_layer_norm = nn.LayerNorm(embedding_dim)
        self.self_attn = DecoderSelfAttention(head_count, embedding_dim)
        self.self_attn_layer_norm = nn.LayerNorm(embedding_dim)
        self.pre_encoder_attn_layer_norm = nn.LayerNorm(embedding_dim)
        self.encoder_attn = DecoderCrossAttention(head_count, embedding_dim)
        self.encoder_attn_layer_norm = nn.LayerNorm(embedding_dim)
        self.glu = GLU(embedding_dim, glu_embedding_dim)


    def forward(
        self,
        decoder_state: FloatTensor,
        encoder_state: FloatTensor,
        attention_state: FloatTensor,
        attention_mask: BoolTensor,
        token_index: LongTensor
    ) -> Tuple[FloatTensor, FloatTensor]:
        # Self Attention
        residual = decoder_state
        decoder_state = self.pre_self_attn_layer_norm.forward(decoder_state)
        decoder_state, attention_state = self.self_attn.forward(
            decoder_state,
            attention_state,
            token_index
        )
        decoder_state = self.self_attn_layer_norm.forward(decoder_state)
        decoder_state = residual + decoder_state

        # Cross Attention
        residual = decoder_state
        decoder_state = self.pre_encoder_attn_layer_norm.forward(decoder_state)
        decoder_state = self.encoder_attn.forward(
            decoder_state,
            encoder_state,
            attention_mask
        )
        decoder_state = self.encoder_attn_layer_norm.forward(decoder_state)
        decoder_state = residual + decoder_state

        # Feed forward
        residual = decoder_state
        decoder_state = self.glu.forward(decoder_state)
        decoder_state = residual + decoder_state

        return decoder_state, attention_state


class DalleBartDecoder(nn.Module):
    def __init__(
        self,
        image_vocab_len: int,
        embedding_dim: int,
        num_attention_heads: int,
        glu_embedding_dim: int,
        depth: int,
        start_token: int
    ):
        super().__init__()
        self.depth = depth
        self.embedding_dim = embedding_dim
        self.embed_tokens = nn.Embedding(image_vocab_len + 1, embedding_dim)
        self.embed_positions = nn.Embedding(IMAGE_TOKEN_COUNT, embedding_dim)
        self.layers: List[DecoderLayer] = nn.ModuleList([
            DecoderLayer(
                num_attention_heads,
                embedding_dim,
                glu_embedding_dim
            ) 
            for _ in range(depth)
        ])
        self.layernorm_embedding = nn.LayerNorm(embedding_dim)
        self.final_ln = nn.LayerNorm(embedding_dim)
        self.lm_head = nn.Linear(embedding_dim, image_vocab_len + 1, bias=False)
        self.zero_prob = torch.zeros([1])
        self.token_indices = torch.arange(IMAGE_TOKEN_COUNT)
        self.start_token = torch.tensor([start_token]).to(torch.long)
        if torch.cuda.is_available():
            self.zero_prob = self.zero_prob.cuda()
            self.token_indices = self.token_indices.cuda()
            self.start_token = self.start_token.cuda()


    def decode_step(
        self,
        log2_supercondition_factor: int,
        attention_mask: BoolTensor,
        encoder_state: FloatTensor,
        attention_state: FloatTensor,
        prev_tokens: LongTensor,
        token_index: LongTensor
    ) -> Tuple[FloatTensor, FloatTensor]:
        image_count = encoder_state.shape[0] // 2
        token_index_batched = token_index[[0] * image_count * 2]
        prev_tokens = prev_tokens[list(range(image_count)) * 2]
        decoder_state = self.embed_tokens.forward(prev_tokens)
        decoder_state += self.embed_positions.forward(token_index_batched)
        decoder_state = self.layernorm_embedding.forward(decoder_state)
        decoder_state = decoder_state[:, None]
        for i in range(self.depth):
            decoder_state, attention_state[i] = self.layers[i].forward(
                decoder_state,
                encoder_state,
                attention_state[i],
                attention_mask,
                token_index
            )
        decoder_state = self.final_ln(decoder_state)
        logits = self.lm_head(decoder_state)
        a = 2 ** log2_supercondition_factor
        logits: FloatTensor = (
            logits[:image_count, -1] * (1 - a) + 
            logits[image_count:, -1] * a
        )

        top_logits, _ = logits.topk(50, dim=-1)
        probs = torch.where(
            logits < top_logits[:, [-1]],
            self.zero_prob,
            torch.exp(logits - top_logits[:, [0]])
        )
        return probs, attention_state


    def decode_row(
        self,
        row_index: int,
        log2_supercondition_factor: int,
        encoder_state: FloatTensor,
        attention_mask: BoolTensor,
        attention_state: FloatTensor,
        image_tokens_sequence: LongTensor
    ) -> Tuple[FloatTensor, LongTensor]:
        for col_index in range(16):
            i = 16 * row_index + col_index
            probs, attention_state = self.decode_step(
                log2_supercondition_factor = log2_supercondition_factor,
                attention_mask = attention_mask,
                encoder_state = encoder_state,
                attention_state = attention_state,
                prev_tokens = image_tokens_sequence[:, i],
                token_index = self.token_indices[[i]]
            )
            image_tokens_sequence[:, i + 1] = torch.multinomial(probs, 1)[:, 0]

        return attention_state, image_tokens_sequence

    
    def decode_initial(
        self,
        seed: int,
        image_count: int,
        text_tokens: LongTensor,
        encoder_state: FloatTensor
    ) -> Tuple[FloatTensor, FloatTensor, FloatTensor, LongTensor]:
        expanded_indices = [0] * image_count + [1] * image_count
        text_tokens = text_tokens[expanded_indices]
        encoder_state = encoder_state[expanded_indices]
        attention_mask = text_tokens.not_equal(1)

        attention_state_shape = (
            self.depth,
            image_count * 4,
            IMAGE_TOKEN_COUNT,
            self.embedding_dim
        )
        attention_state = torch.zeros(attention_state_shape)
        image_tokens_sequence = torch.full(
            (image_count, IMAGE_TOKEN_COUNT + 1), 
            BLANK_TOKEN,
            dtype=torch.long
        )
        if torch.cuda.is_available(): 
            attention_state = attention_state.cuda()
            image_tokens_sequence = image_tokens_sequence.cuda()
        
        image_tokens_sequence[:, 0] = self.start_token[0]

        if seed > 0: torch.manual_seed(seed)

        return encoder_state, attention_mask, attention_state, image_tokens_sequence

import torch
from torch import Tensor
from torch.nn import Module, ModuleList, GroupNorm, Conv2d, Embedding
torch.set_grad_enabled(False)


class ResnetBlock(Module):
    def __init__(self, log2_count_in: int, log2_count_out: int):
        super().__init__()
        m, n = 2 ** log2_count_in, 2 ** log2_count_out
        self.is_middle = m == n
        self.norm1 = GroupNorm(2 ** 5, m)
        self.conv1 = Conv2d(m, n, 3, padding=1)
        self.norm2 = GroupNorm(2 ** 5, n)
        self.conv2 = Conv2d(n, n, 3, padding=1)
        if not self.is_middle:
            self.nin_shortcut = Conv2d(m, n, 1)

    def forward(self, x: Tensor) -> Tensor:
        h = x
        h = self.norm1.forward(h)
        h *= torch.sigmoid(h)
        h = self.conv1.forward(h)
        h = self.norm2.forward(h)
        h *= torch.sigmoid(h)
        h = self.conv2(h)
        if not self.is_middle:
            x = self.nin_shortcut.forward(x)
        return x + h


class AttentionBlock(Module):
    def __init__(self):
        super().__init__()
        n = 2 ** 9
        self.norm = GroupNorm(2 ** 5, n)
        self.q = Conv2d(n, n, 1)
        self.k = Conv2d(n, n, 1)
        self.v = Conv2d(n, n, 1)
        self.proj_out = Conv2d(n, n, 1)

    def forward(self, x: Tensor) -> Tensor:
        n, m = 2 ** 9, x.shape[0]
        h = x
        h = self.norm(h)
        q = self.q.forward(h)
        k = self.k.forward(h)
        v = self.v.forward(h)
        q = q.reshape(m, n, 2 ** 8)
        q = q.permute(0, 2, 1)
        k = k.reshape(m, n, 2 ** 8)
        w = torch.bmm(q, k)
        w /= n ** 0.5
        w = torch.softmax(w, dim=2)
        v = v.reshape(m, n, 2 ** 8)
        w = w.permute(0, 2, 1)
        h = torch.bmm(v, w)
        h = h.reshape(m, n, 2 ** 4, 2 ** 4)
        h = self.proj_out.forward(h)
        return x + h


class MiddleLayer(Module):
    def __init__(self):
        super().__init__()
        self.block_1 = ResnetBlock(9, 9)
        self.attn_1 = AttentionBlock()
        self.block_2 = ResnetBlock(9, 9)
    
    def forward(self, h: Tensor) -> Tensor:
        h = self.block_1.forward(h)
        h = self.attn_1.forward(h)
        h = self.block_2.forward(h)
        return h


class Upsample(Module):
    def __init__(self, log2_count):
        super().__init__()
        n = 2 ** log2_count
        self.upsample = torch.nn.UpsamplingNearest2d(scale_factor=2)
        self.conv = Conv2d(n, n, 3, padding=1)

    def forward(self, x: Tensor) -> Tensor:
        x = self.upsample.forward(x)
        x = self.conv.forward(x)
        return x


class UpsampleBlock(Module):
    def __init__(
        self, 
        log2_count_in: int, 
        log2_count_out: int, 
        has_attention: bool, 
        has_upsample: bool
    ):
        super().__init__()
        self.has_attention = has_attention
        self.has_upsample = has_upsample
        self.block = ModuleList([
            ResnetBlock(log2_count_in, log2_count_out),
            ResnetBlock(log2_count_out, log2_count_out),
            ResnetBlock(log2_count_out, log2_count_out)
        ])
        if has_attention:
            self.attn = ModuleList([
                AttentionBlock(),
                AttentionBlock(),
                AttentionBlock()
            ])
        else:
            self.attn = ModuleList()

        if has_upsample:
            self.upsample = Upsample(log2_count_out)


    def forward(self, h: Tensor) -> Tensor:
        for j in range(3):
            h = self.block[j].forward(h)
            if self.has_attention:
                h = self.attn[j].forward(h)
        if self.has_upsample:
            h = self.upsample.forward(h)
        return h


class Decoder(Module):
    def __init__(self):
        super().__init__()

        self.conv_in = Conv2d(2 ** 8, 2 ** 9, 3, padding=1)
        self.mid = MiddleLayer()

        self.up = ModuleList([
            UpsampleBlock(7, 7, False, False),
            UpsampleBlock(8, 7, False, True),
            UpsampleBlock(8, 8, False, True),
            UpsampleBlock(9, 8, False, True),
            UpsampleBlock(9, 9, True, True)
        ])

        self.norm_out = GroupNorm(2 ** 5, 2 ** 7)
        self.conv_out = Conv2d(2 ** 7, 3, 3, padding=1)

    def forward(self, z: Tensor) -> Tensor:
        z = self.conv_in.forward(z)
        z = self.mid.forward(z)

        for i in reversed(range(5)):
            z = self.up[i].forward(z)

        z = self.norm_out.forward(z)
        z *= torch.sigmoid(z)
        z = self.conv_out.forward(z)
        return z


class VQGanDetokenizer(Module):
    def __init__(self):
        super().__init__()
        m, n = 2 ** 14, 2 ** 8
        self.embedding = Embedding(m, n)
        self.post_quant_conv = Conv2d(n, n, 1)
        self.decoder = Decoder()

    def forward(self, z: Tensor) -> Tensor:
        z = self.embedding.forward(z)
        z = z.view((z.shape[0], 2 ** 4, 2 ** 4, 2 ** 8))
        z = z.permute(0, 3, 1, 2).contiguous()
        z = self.post_quant_conv.forward(z)
        z = self.decoder.forward(z)
        z = z.permute(0, 2, 3, 1)
        z = z.clip(0.0, 1.0) * 255
        return z

import os
from PIL import Image
import numpy
from torch import LongTensor
import torch
import json
import requests
from typing import Iterator
torch.set_grad_enabled(False)
# torch.set_num_threads(os.cpu_count())

class MinDalle(nn.Module):
    def __init__(
        self,
        args,
        is_reusable: bool = True,
        models_root: str = 'pretrained',
    ):
        super().__init__()
        self.is_reusable = is_reusable
        self.text_token_count = 64
        self.depth = args["depth"]
        self.num_attention_heads = args["num_attention_heads"]
        self.embedding_dim = args["embedding_dim"]
        self.glu_embedding_dim = args["glu_embedding_dim"]
        self.text_vocab_len = args["text_vocab_len"]
        self.image_vocab_len = args["image_vocab_len"]
        with open(args["vocab_path"], 'r', encoding='utf8') as f:
            vocab = json.load(f)
        with open(args["merges_path"], 'r', encoding='utf8') as f:
            merges = f.read().split("\n")[1:-1]
        self.tokenizer = TextTokenizer(vocab, merges)
        self.encoder = DalleBartEncoder(
            num_attention_heads = self.num_attention_heads,
            embedding_dim = self.embedding_dim,
            glu_embedding_dim = self.glu_embedding_dim,
            text_token_count = self.text_token_count,
            text_vocab_len = self.text_vocab_len,
            depth = self.depth
        )
        self.decoder = DalleBartDecoder(
            image_vocab_len = self.image_vocab_len,
            num_attention_heads = self.num_attention_heads,
            embedding_dim = self.embedding_dim,
            glu_embedding_dim = self.glu_embedding_dim,
            depth = self.depth,
            start_token = self.image_vocab_len
        )
        self.detokenizer = VQGanDetokenizer()
        self.device = args["device"]


    def image_from_tokens(
        self,
        grid_size: int,
        image_tokens: LongTensor,
        is_verbose: bool = False
    ) -> Image.Image:
        if not self.is_reusable: del self.decoder
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        if not self.is_reusable: self.init_detokenizer()
        if is_verbose: print("detokenizing image")
        images = self.detokenizer.forward(image_tokens).to(torch.uint8)
        if not self.is_reusable: del self.detokenizer
        all_images = []
        for i in range(grid_size ** 2):
            image = images[i, :] # [256, 256, 3]
            # image = image.flatten(1, 2).transpose(0, 1).flatten(1, 2)
            all_images.append(Image.fromarray(image.to('cpu').detach().numpy()))
        return all_images


    def generate_image_stream(
        self, 
        text: str, 
        seed: int,
        grid_size: int,
        log2_mid_count: int,
        log2_supercondition_factor: int = 3,
        is_verbose: bool = False
    ) -> Iterator[Image.Image]:
        assert(log2_mid_count in range(5))
        if is_verbose: print("tokenizing text")
        tokens = self.tokenizer.tokenize(text.lower(), is_verbose=is_verbose)
        if is_verbose: print("text tokens", tokens)
        text_tokens = numpy.ones((2, 64), dtype=numpy.int32)
        text_tokens[0, :2] = [tokens[0], tokens[-1]]
        text_tokens[1, :len(tokens)] = tokens

        text_tokens = torch.tensor(text_tokens).to(torch.long)
        text_tokens = text_tokens.to(self.device)

        if not self.is_reusable: self.init_encoder()
        if is_verbose: print("encoding text tokens")
        encoder_state = self.encoder.forward(text_tokens)
        if not self.is_reusable: del self.encoder
        if torch.cuda.is_available(): torch.cuda.empty_cache()

        if not self.is_reusable: self.init_decoder()

        encoder_state, attention_mask, attention_state, image_tokens = ( 
            self.decoder.decode_initial(
                seed, 
                grid_size ** 2, 
                text_tokens, 
                encoder_state
            )
        )

        row_count = 16
        for row_index in range(row_count):
            if is_verbose: 
                print('sampling row {} of {}'.format(row_index + 1, row_count))
            attention_state, image_tokens = self.decoder.decode_row(
                row_index,
                log2_supercondition_factor,
                encoder_state,
                attention_mask,
                attention_state,
                image_tokens
            )
            if ((row_index + 1) * (2 ** log2_mid_count)) % row_count == 0:
                tokens = image_tokens[:, 1:]
                image = self.image_from_tokens(grid_size, tokens, is_verbose)

        return image

    def forward(
        self, 
        text: str,
        seed: int = -1,
        grid_size: int = 1,
        log2_supercondition_factor: int = 3,
        is_verbose: bool = False
    ) -> Image.Image:
        log2_mid_count = 0
        image = self.generate_image_stream(
            text,
            seed,
            grid_size,
            log2_mid_count,
            log2_supercondition_factor,
            is_verbose
        )
        return image