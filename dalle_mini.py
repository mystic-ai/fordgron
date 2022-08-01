from fordgron import DALLE
import torch
from os.path import exists
import requests
import json
from transformers import BartTokenizerFast

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

model = DALLE(model_kwargs).to(model_kwargs["device"])

model.encoder.load_state_dict(
    torch.load("./will-dalle-files/mini/encoder.pt"), strict=False
)

model.decoder.load_state_dict(
    torch.load("./will-dalle-files/mini/decoder.pt"), strict=False
)

model.detokenizer.load_state_dict(torch.load("./will-dalle-files/detokenizer.pt"))

text = "george washington eating battenburg cake anime"

tokenizer = BartTokenizerFast(
    vocab_file="./will-dalle-files/mini/vocab.json",
    merges_file="./will-dalle-files/mini/merges.txt",
    sep_token="</s>",
    cls_token="<s>",
    unk_token="<unk>",
)

input = tokenizer.encode(chr(ord(" ")) + text.lower())
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
