from typing import List
import torch
from torch import nn, BoolTensor, FloatTensor, LongTensor


class GLU(nn.Module):
    def __init__(self, count_in_out: int, count_middle: int):
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

    def forward(
        self,
        keys: FloatTensor,
        values: FloatTensor,
        queries: FloatTensor,
        attention_mask: BoolTensor,
    ) -> FloatTensor:
        keys = keys.reshape(keys.shape[:2] + (self.head_count, -1))
        values = values.reshape(values.shape[:2] + (self.head_count, -1))
        queries = queries.reshape(queries.shape[:2] + (self.head_count, -1))
        queries /= queries.shape[-1] ** 0.5

        attention_bias = (1 - attention_mask.to(torch.float32)) * -1e12
        attention_weights: FloatTensor = torch.einsum("bqhc,bkhc->bhqk", queries, keys)
        attention_weights += attention_bias[:, None, None, :]
        attention_weights = torch.softmax(attention_weights, -1)
        attention_output: FloatTensor = torch.einsum(
            "bhqk,bkhc->bqhc", attention_weights, values
        )
        shape = attention_output.shape[:2] + (self.embedding_dim,)
        attention_output = attention_output.reshape(shape)
        attention_output = self.out_proj.forward(attention_output)
        return attention_output


class EncoderSelfAttention(AttentionBase):
    def forward(
        self, encoder_state: FloatTensor, attention_mask: BoolTensor
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
        self, encoder_state: FloatTensor, attention_mask: BoolTensor
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
        layer_count: int,
        embedding_dim: int,
        num_attention_heads: int,
        text_vocab_len: int,
        text_token_count: int,
        glu_embedding_dim: int,
        device: str,
    ):
        super().__init__()
        self.text_vocab_len = text_vocab_len
        self.embed_tokens = nn.Embedding(text_vocab_len, embedding_dim)
        self.embed_positions = nn.Embedding(text_token_count, embedding_dim)
        self.layers: List[EncoderLayer] = nn.ModuleList(
            [
                EncoderLayer(
                    embedding_dim=embedding_dim,
                    head_count=num_attention_heads,
                    glu_embedding_dim=glu_embedding_dim,
                )
                for _ in range(layer_count)
            ]
        )
        self.layernorm_embedding = nn.LayerNorm(embedding_dim)
        self.final_ln = nn.LayerNorm(embedding_dim)
        token_indices = torch.arange(text_token_count, device=device)
        self.pose_tokens = torch.stack([token_indices] * 2)

    def forward(self, text_tokens: LongTensor) -> FloatTensor:
        attention_mask = text_tokens.not_equal(1)
        encoder_state = self.embed_tokens.forward(
            text_tokens
        ) + self.embed_positions.forward(self.pose_tokens)
        encoder_state = self.layernorm_embedding.forward(encoder_state)
        for layer in self.layers:
            encoder_state = layer.forward(encoder_state, attention_mask)
        encoder_state = self.final_ln.forward(encoder_state)
        return encoder_state


import torch
from torch import nn
from torch import FloatTensor, LongTensor
from math import sqrt


class ResnetBlock(nn.Module):
    def __init__(self, log2_count_in: int, log2_count_out: int):
        super().__init__()
        m, n = 2**log2_count_in, 2**log2_count_out
        self.is_middle = m == n
        self.norm1 = nn.GroupNorm(2**5, m)
        self.conv1 = nn.Conv2d(m, n, 3, padding=1)
        self.norm2 = nn.GroupNorm(2**5, n)
        self.conv2 = nn.Conv2d(n, n, 3, padding=1)
        if not self.is_middle:
            self.nin_shortcut = nn.Conv2d(m, n, 1)

    def forward(self, x: FloatTensor) -> FloatTensor:
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


class AttentionBlock(nn.Module):
    def __init__(self):
        super().__init__()
        n = 2**9
        self.norm = nn.GroupNorm(2**5, n)
        self.q = nn.Conv2d(n, n, 1)
        self.k = nn.Conv2d(n, n, 1)
        self.v = nn.Conv2d(n, n, 1)
        self.proj_out = nn.Conv2d(n, n, 1)

    def forward(self, x: FloatTensor) -> FloatTensor:
        n, m = 2**9, x.shape[0]
        h = x
        h = self.norm(h)
        k = self.k.forward(h)
        v = self.v.forward(h)
        q = self.q.forward(h)
        k = k.reshape(m, n, -1)
        v = v.reshape(m, n, -1)
        q = q.reshape(m, n, -1)
        q = q.permute(0, 2, 1)
        w = torch.bmm(q, k)
        w /= n**0.5
        w = torch.softmax(w, dim=2)
        w = w.permute(0, 2, 1)
        h = torch.bmm(v, w)
        token_count = int(sqrt(h.shape[-1]))
        h = h.reshape(m, n, token_count, token_count)
        h = self.proj_out.forward(h)
        return x + h


class MiddleLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.block_1 = ResnetBlock(9, 9)
        self.attn_1 = AttentionBlock()
        self.block_2 = ResnetBlock(9, 9)

    def forward(self, h: FloatTensor) -> FloatTensor:
        h = self.block_1.forward(h)
        h = self.attn_1.forward(h)
        h = self.block_2.forward(h)
        return h


class Upsample(nn.Module):
    def __init__(self, log2_count):
        super().__init__()
        n = 2**log2_count
        self.upsample = torch.nn.UpsamplingNearest2d(scale_factor=2)
        self.conv = nn.Conv2d(n, n, 3, padding=1)

    def forward(self, x: FloatTensor) -> FloatTensor:
        x = self.upsample.forward(x.to(torch.float32))
        x = self.conv.forward(x)
        return x


class UpsampleBlock(nn.Module):
    def __init__(
        self,
        log2_count_in: int,
        log2_count_out: int,
        has_attention: bool,
        has_upsample: bool,
    ):
        super().__init__()
        self.has_attention = has_attention
        self.has_upsample = has_upsample

        self.block = nn.ModuleList(
            [
                ResnetBlock(log2_count_in, log2_count_out),
                ResnetBlock(log2_count_out, log2_count_out),
                ResnetBlock(log2_count_out, log2_count_out),
            ]
        )

        if has_attention:
            self.attn = nn.ModuleList(
                [AttentionBlock(), AttentionBlock(), AttentionBlock()]
            )

        if has_upsample:
            self.upsample = Upsample(log2_count_out)

    def forward(self, h: FloatTensor) -> FloatTensor:
        for j in range(3):
            h = self.block[j].forward(h)
            if self.has_attention:
                h = self.attn[j].forward(h)
        if self.has_upsample:
            h = self.upsample.forward(h)
        return h


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv_in = nn.Conv2d(2**8, 2**9, 3, padding=1)
        self.mid = MiddleLayer()

        self.up = nn.ModuleList(
            [
                UpsampleBlock(7, 7, False, False),
                UpsampleBlock(8, 7, False, True),
                UpsampleBlock(8, 8, False, True),
                UpsampleBlock(9, 8, False, True),
                UpsampleBlock(9, 9, True, True),
            ]
        )

        self.norm_out = nn.GroupNorm(2**5, 2**7)
        self.conv_out = nn.Conv2d(2**7, 3, 3, padding=1)

    def forward(self, z: FloatTensor) -> FloatTensor:
        z = self.conv_in.forward(z)
        z = self.mid.forward(z)

        for i in reversed(range(5)):
            z = self.up[i].forward(z)

        z = self.norm_out.forward(z)
        z *= torch.sigmoid(z)
        z = self.conv_out.forward(z)
        return z


class VQGanDetokenizer(nn.Module):
    def __init__(self):
        super().__init__()
        vocab_count, embedding_dim = 2**14, 2**8
        self.vocab_count = vocab_count
        self.embedding = nn.Embedding(vocab_count, embedding_dim)
        self.post_quant_conv = nn.Conv2d(embedding_dim, embedding_dim, 1)
        self.decoder = Decoder()

    def forward(self, is_seamless: bool, z: LongTensor) -> FloatTensor:
        z.clamp_(0, self.vocab_count - 1)
        grid_size = int(sqrt(z.shape[0]))
        token_count = grid_size * 2**4

        if is_seamless:
            z = z.view([grid_size, grid_size, 2**4, 2**4])
            z = z.flatten(1, 2).transpose(1, 0).flatten(1, 2)
            z = z.flatten().unsqueeze(1)
            z = self.embedding.forward(z)
            z = z.view((1, token_count, token_count, 2**8))
        else:
            z = self.embedding.forward(z)
            z = z.view((z.shape[0], 2**4, 2**4, 2**8))

        z = z.permute(0, 3, 1, 2).contiguous()
        z = self.post_quant_conv.forward(z)
        z = self.decoder.forward(z)
        z = z.permute(0, 2, 3, 1)
        z = z.clip(0.0, 1.0) * 255

        if is_seamless:
            z = z[0]
        else:
            z = z.view([grid_size, grid_size, 2**8, 2**8, 3])
            z = z.flatten(1, 2).transpose(1, 0).flatten(1, 2)

        return z


from typing import Tuple, List
import torch
from torch import nn, LongTensor, FloatTensor, BoolTensor

IMAGE_TOKEN_COUNT = 256


class DecoderCrossAttention(AttentionBase):
    def forward(
        self,
        decoder_state: FloatTensor,
        encoder_state: FloatTensor,
        attention_mask: BoolTensor,
    ) -> FloatTensor:
        keys = self.k_proj.forward(encoder_state)
        values = self.v_proj.forward(encoder_state)
        queries = self.q_proj.forward(decoder_state)
        return super().forward(keys, values, queries, attention_mask)


class DecoderSelfAttention(AttentionBase):
    def __init__(self, head_count: int, embedding_dim: int):
        super().__init__(head_count, embedding_dim)

    def forward(
        self,
        decoder_state: FloatTensor,
        attention_state: FloatTensor,
        attn_mask: BoolTensor,
        token_index: LongTensor,
    ) -> Tuple[FloatTensor, FloatTensor]:
        keys = self.k_proj.forward(decoder_state)
        values = self.v_proj.forward(decoder_state)
        queries = self.q_proj.forward(decoder_state)
        attn_state_new = torch.cat([keys, values]).to(attention_state.dtype)
        attention_state[:, token_index] = attn_state_new
        batch_count = decoder_state.shape[0]
        keys = attention_state[:batch_count]
        values = attention_state[batch_count:]
        decoder_state = super().forward(keys, values, queries, attn_mask)
        return decoder_state, attention_state


class DecoderLayer(nn.Module):
    def __init__(
        self, head_count: int, embedding_dim: int, glu_embedding_dim: int, device: str
    ):
        super().__init__()
        self.pre_self_attn_layer_norm = nn.LayerNorm(embedding_dim)
        self.self_attn = DecoderSelfAttention(head_count, embedding_dim)
        self.self_attn_layer_norm = nn.LayerNorm(embedding_dim)
        self.pre_encoder_attn_layer_norm = nn.LayerNorm(embedding_dim)
        self.encoder_attn = DecoderCrossAttention(head_count, embedding_dim)
        self.encoder_attn_layer_norm = nn.LayerNorm(embedding_dim)
        self.glu = GLU(embedding_dim, glu_embedding_dim)
        self.token_indices = torch.arange(IMAGE_TOKEN_COUNT, device=device)

    def forward(
        self,
        decoder_state: FloatTensor,
        encoder_state: FloatTensor,
        attention_state: FloatTensor,
        attention_mask: BoolTensor,
        token_index: LongTensor,
    ) -> Tuple[FloatTensor, FloatTensor]:
        # Self Attention
        self_attn_mask = self.token_indices < token_index + 1
        self_attn_mask = self_attn_mask[None][[0] * decoder_state.shape[0]]
        residual = decoder_state
        decoder_state = self.pre_self_attn_layer_norm.forward(decoder_state)
        decoder_state, attention_state = self.self_attn.forward(
            decoder_state=decoder_state,
            attention_state=attention_state,
            attn_mask=self_attn_mask,
            token_index=token_index,
        )
        decoder_state = self.self_attn_layer_norm.forward(decoder_state)
        decoder_state = residual + decoder_state

        # Cross Attention
        residual = decoder_state
        decoder_state = self.pre_encoder_attn_layer_norm.forward(decoder_state)
        decoder_state = self.encoder_attn.forward(
            decoder_state=decoder_state,
            encoder_state=encoder_state,
            attention_mask=attention_mask,
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
        layer_count: int,
        device: str,
    ):
        super().__init__()
        self.layer_count = layer_count
        self.embedding_dim = embedding_dim
        self.image_vocab_len = image_vocab_len
        self.embed_tokens = nn.Embedding(image_vocab_len + 1, embedding_dim)
        self.embed_positions = nn.Embedding(IMAGE_TOKEN_COUNT, embedding_dim)
        self.layers: List[DecoderLayer] = nn.ModuleList(
            [
                DecoderLayer(
                    head_count=num_attention_heads,
                    embedding_dim=embedding_dim,
                    glu_embedding_dim=glu_embedding_dim,
                    device=device,
                )
                for _ in range(layer_count)
            ]
        )
        self.layernorm_embedding = nn.LayerNorm(embedding_dim)
        self.final_ln = nn.LayerNorm(embedding_dim)
        self.lm_head = nn.Linear(embedding_dim, image_vocab_len + 1, bias=False)
        self.token_indices = torch.arange(IMAGE_TOKEN_COUNT, device=device)

    def forward(
        self,
        settings: FloatTensor,
        attention_mask: BoolTensor,
        encoder_state: FloatTensor,
        attention_state: FloatTensor,
        prev_tokens: LongTensor,
        token_index: LongTensor,
    ) -> Tuple[LongTensor, FloatTensor]:
        image_count = encoder_state.shape[0] // 2
        token_index_batched = token_index[[0] * image_count * 2]
        prev_tokens = prev_tokens[list(range(image_count)) * 2]
        prev_tokens.clamp_(0, self.image_vocab_len)
        decoder_state = self.embed_tokens.forward(prev_tokens)
        decoder_state += self.embed_positions.forward(token_index_batched)
        decoder_state = self.layernorm_embedding.forward(decoder_state)
        decoder_state = decoder_state[:, None]
        for i in range(self.layer_count):
            decoder_state, attention_state[i] = self.layers[i].forward(
                decoder_state,
                encoder_state,
                attention_state[i],
                attention_mask,
                token_index,
            )
        decoder_state = self.final_ln(decoder_state)
        logits = self.lm_head(decoder_state)
        temperature = settings[[0]]
        top_k = settings[[1]].to(torch.long)
        supercondition_factor = 2 ** settings[[2]]
        logits = logits[:, -1, : 2**14]
        logits: FloatTensor = (
            logits[:image_count] * (1 - supercondition_factor)
            + logits[image_count:] * supercondition_factor
        )
        logits_sorted, _ = logits.sort(descending=True)
        is_kept = logits >= logits_sorted[:, top_k - 1]
        logits -= logits_sorted[:, [0]]
        logits /= temperature
        logits.exp_()
        logits *= is_kept.to(torch.float32)
        image_tokens = torch.multinomial(logits, 1)[:, 0]
        return image_tokens, attention_state


from math import inf
from typing import List, Tuple
from emoji import demojize


class TextTokenizer:
    def __init__(self, vocab: dict, merges: List[str]):
        self.token_from_subword = vocab
        pairs = [tuple(pair.split()) for pair in merges]
        self.rank_from_pair = dict(zip(pairs, range(len(pairs))))

    def tokenize(self, text: str, is_verbose: bool = False) -> List[int]:
        sep_token = self.token_from_subword["</s>"]
        cls_token = self.token_from_subword["<s>"]
        unk_token = self.token_from_subword["<unk>"]
        text = demojize(text, delimiters=["", ""])
        text = text.lower().encode("ascii", errors="ignore").decode()
        tokens = [
            self.token_from_subword.get(subword, unk_token)
            for word in text.split(" ")
            if len(word) > 0
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
            if pair_to_merge not in self.rank_from_pair:
                break
            i = pairs.index(pair_to_merge)
            subwords = (
                (subwords[:i] if i > 0 else [])
                + [subwords[i] + subwords[i + 1]]
                + (subwords[i + 2 :] if i + 2 < len(subwords) else [])
            )

        if is_verbose:
            print(subwords)
        return subwords


import os
from PIL import Image
import numpy
from torch import LongTensor, FloatTensor
import torch
import torch.backends.cudnn, torch.backends.cuda
import json
import requests
from typing import Iterator

torch.set_grad_enabled(False)
torch.set_num_threads(os.cpu_count())
torch.backends.cudnn.enabled = True
torch.backends.cudnn.allow_tf32 = True

MIN_DALLE_REPO = "https://huggingface.co/kuprel/min-dalle/resolve/main/"
IMAGE_TOKEN_COUNT = 256


class DALLE(nn.Module):
    def __init__(
        self,
        args,
        is_reusable: bool = True,
    ):
        super().__init__()
        if args["device"] == None:
            args["device"] = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = args["device"]
        self.text_token_count = 64
        self.layer_count = args["depth"]
        self.num_attention_heads = args["num_attention_heads"]
        self.embedding_dim = args["embedding_dim"]
        self.glu_embedding_dim = args["glu_embedding_dim"]
        self.text_vocab_len = args["text_vocab_len"]
        self.image_vocab_len = args["image_vocab_len"]

        self.is_verbose = False
        self.is_mega = False
        self.dtype = torch.float32
        self.is_reusable = True
        with open(args["vocab_path"], "r", encoding="utf8") as f:
            vocab = json.load(f)
        with open(args["merges_path"], "r", encoding="utf8") as f:
            merges = f.read().split("\n")[1:-1]
        self.tokenizer = TextTokenizer(vocab, merges)
        self.encoder = (
            DalleBartEncoder(
                num_attention_heads=self.num_attention_heads,
                embedding_dim=self.embedding_dim,
                glu_embedding_dim=self.glu_embedding_dim,
                text_token_count=self.text_token_count,
                text_vocab_len=self.text_vocab_len,
                layer_count=self.layer_count,
                device=self.device,
            )
            .to(self.dtype)
            .eval()
        )
        self.decoder = (
            DalleBartDecoder(
                image_vocab_len=self.image_vocab_len,
                num_attention_heads=self.num_attention_heads,
                embedding_dim=self.embedding_dim,
                glu_embedding_dim=self.glu_embedding_dim,
                layer_count=self.layer_count,
                device=self.device,
            )
            .to(self.dtype)
            .eval()
        )
        self.detokenizer = VQGanDetokenizer().eval()

    def image_grid_from_tokens(
        self, image_tokens: LongTensor, is_seamless: bool, is_verbose: bool = False
    ) -> FloatTensor:
        if not self.is_reusable:
            del self.decoder
        torch.cuda.empty_cache()
        if not self.is_reusable:
            self.init_detokenizer()
        if is_verbose:
            print("detokenizing image")
        images = self.detokenizer.forward(is_seamless, image_tokens)  # [768, 768, 3]
        if not self.is_reusable:
            del self.detokenizer
        return images

    def generate_raw_image_stream(
        self,
        text: str,
        seed: int,
        grid_size: int,
        progressive_outputs: bool = False,
        is_seamless: bool = False,
        temperature: float = 1,
        top_k: int = 256,
        log2_supercondition_factor: int = 4,
        is_verbose: bool = False,
    ) -> Iterator[FloatTensor]:
        image_count = grid_size**2
        if is_verbose:
            print("tokenizing text")
        tokens = self.tokenizer.tokenize(text, is_verbose=is_verbose)
        if len(tokens) > self.text_token_count:
            tokens = tokens[: self.text_token_count]
        if is_verbose:
            print("{} text tokens".format(len(tokens)), tokens)
        text_tokens = numpy.ones((2, 64), dtype=numpy.int32)
        text_tokens[0, :2] = [tokens[0], tokens[-1]]
        text_tokens[1, : len(tokens)] = tokens
        text_tokens = torch.tensor(text_tokens, dtype=torch.long, device=self.device)

        if not self.is_reusable:
            self.init_encoder()
        if is_verbose:
            print("encoding text tokens")
        with torch.cuda.amp.autocast(dtype=self.dtype):
            encoder_state = self.encoder.forward(text_tokens)  # [2, 64, 1024]
        if not self.is_reusable:
            del self.encoder
        torch.cuda.empty_cache()

        if not self.is_reusable:
            self.init_decoder()

        with torch.cuda.amp.autocast(dtype=self.dtype):
            expanded_indices = [0] * image_count + [1] * image_count
            text_tokens = text_tokens[expanded_indices]
            encoder_state = encoder_state[expanded_indices]  # [18, 64, 1024]
            attention_mask = text_tokens.not_equal(1)
            attention_state = torch.zeros(
                size=(
                    self.layer_count,
                    image_count * 4,
                    IMAGE_TOKEN_COUNT,
                    self.embedding_dim,
                ),
                device=self.device,
            )
            image_tokens = torch.full(
                (IMAGE_TOKEN_COUNT + 1, image_count),
                self.image_vocab_len,
                dtype=torch.long,
                device=self.device,
            )

            if seed > 0:
                torch.manual_seed(seed)

        token_indices = torch.arange(IMAGE_TOKEN_COUNT, device=self.device)
        settings = torch.tensor(
            [temperature, top_k, log2_supercondition_factor],
            dtype=torch.float32,
            device=self.device,
        )
        for i in range(IMAGE_TOKEN_COUNT):
            torch.cuda.empty_cache()
            with torch.cuda.amp.autocast(dtype=self.dtype):
                image_tokens[i + 1], attention_state = self.decoder.forward(
                    settings=settings,
                    attention_mask=attention_mask,
                    encoder_state=encoder_state,
                    attention_state=attention_state,
                    prev_tokens=image_tokens[i],
                    token_index=token_indices[[i]],
                )

            with torch.cuda.amp.autocast(dtype=torch.float32):
                if ((i + 1) % 32 == 0 and progressive_outputs) or i + 1 == 256:
                    yield self.image_grid_from_tokens(
                        image_tokens=image_tokens[1:].T,
                        is_seamless=is_seamless,
                        is_verbose=is_verbose,
                    )

    def forward(
        self,
        text: str,
        seed: int = -1,
        grid_size: int = 1,
        log2_supercondition_factor: int = 4,
        grid: bool = False,
    ) -> Image.Image:
        log2_mid_count = 0
        all_images = []
        image_stream = self.generate_raw_image_stream(
            text=text,
            seed=seed,
            grid_size=grid_size,
            log2_supercondition_factor=log2_supercondition_factor,
        )
        if grid:
            for image in image_stream:
                return [
                    Image.fromarray(image.to(torch.uint8).to("cpu").detach().numpy())
                ]
        else:
            all_images = []
            for image in image_stream:
                image = image.view(
                    [grid_size * IMAGE_TOKEN_COUNT, grid_size, IMAGE_TOKEN_COUNT, -1]
                )
                image = image.transpose(1, 0)
                image = image.reshape(
                    [grid_size**2, IMAGE_TOKEN_COUNT, IMAGE_TOKEN_COUNT, -1]
                )
                for i in range(grid_size**2):
                    selected_image = image[i, :]
                    all_images.append(
                        Image.fromarray(
                            selected_image.to(torch.uint8).to("cpu").detach().numpy()
                        )
                    )
            return all_images
