from fordgron import DALLE
import torch
from os.path import exists
import requests
from typing import List, Tuple
from emoji import demojize
from math import inf
import json

required_files = [
    (
        "https://huggingface.co/kuprel/min-dalle/resolve/main/encoder_mini.pt",
        "./will-dalle-files/mini/encoder.pt",
    ),
    (
        "https://huggingface.co/kuprel/min-dalle/resolve/main/decoder_mini.pt",
        "./will-dalle-files/mini/decoder.pt",
    ),
    (
        "https://huggingface.co/kuprel/min-dalle/resolve/main/vocab_mini.json",
        "./will-dalle-files/mini/vocab.json",
    ),
    (
        "https://huggingface.co/kuprel/min-dalle/resolve/main/merges_mini.json",
        "./will-dalle-files/mini/merges.txt",
    ),
    (
        "https://huggingface.co/kuprel/min-dalle/resolve/main/detoker.pt",
        "./will-dalle-files/detokenizer.pt",
    ),
]


class TextTokenizer:
    def __init__(self, vocab: dict, merges: List[str]):
        self.token_from_subword = vocab
        pairs = [tuple(pair.split()) for pair in merges]
        self.rank_from_pair = dict(zip(pairs, range(len(pairs))))

    def tokenize(self, text: str) -> List[int]:
        sep_token = self.token_from_subword["</s>"]
        cls_token = self.token_from_subword["<s>"]
        unk_token = self.token_from_subword["<unk>"]
        text = demojize(text, delimiters=["", ""])
        text = text.lower().encode("ascii", errors="ignore").decode()
        tokens = [
            self.token_from_subword.get(subword, unk_token)
            for word in text.split(" ")
            if len(word) > 0
            for subword in self.get_byte_pair_encoding(word)
        ]
        return [cls_token] + tokens + [sep_token]

    def get_byte_pair_encoding(self, word: str) -> List[str]:
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
        return subwords


def download_file(url: str, save_path: str):
    print(f"downloading {url}")
    download = requests.get(url)
    with open(save_path, "wb") as f:
        f.write(download.content)


for file in required_files:
    if not exists(file[1]):
        download_file(file[0], file[1])

model_kwargs = {
    "depth": 12,
    "num_attention_heads": 16,
    "embedding_dim": 1024,
    "glu_embedding_dim": 2730,
    "max_input_text_tokens": 64,
    "text_vocab_len": 50264,
    "image_vocab_len": 16384,
    "device": 0,
    "dtype": torch.float16,
}

with open("./will-dalle-files/mini/vocab.json", "r", encoding="utf8") as f:
    vocab = json.load(f)
with open("./will-dalle-files/mini/merges.txt", "r", encoding="utf8") as f:
    merges = f.read().split("\n")[1:-1]
tokenizer = TextTokenizer(
    vocab,
    merges,
)

model = DALLE(model_kwargs).to(model_kwargs["device"])

model.encoder.load_state_dict(
    torch.load("./will-dalle-files/mini/encoder.pt"), strict=False
)

model.decoder.load_state_dict(
    torch.load("./will-dalle-files/mini/decoder.pt"), strict=False
)

model.detokenizer.load_state_dict(torch.load("./will-dalle-files/detokenizer.pt"))

text = "blob in a horse house"

input = tokenizer.tokenize(text)
if len(input) > model_kwargs["max_input_text_tokens"]:
    print(
        f"too many input tokens, trucating to the last {model_kwargs['max_input_text_tokens']}"
    )
    input = input[-model_kwargs["max_input_text_tokens"] :]

images = model(
    input,
    seed=-1,
    grid_size=3,
    grid=True,
)

for i, image in enumerate(images):
    image.save(f"image_{i}.png")
