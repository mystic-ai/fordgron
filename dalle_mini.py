from fordgron import DALLE
import time
import torch
from os.path import exists
import requests

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
    "text_vocab_len": 50264,
    "image_vocab_len": 16384,
    "vocab_path": "./will-dalle-files/mini/vocab.json",
    "merges_path": "./will-dalle-files/mini/merges.txt",
    "device": 0,
}

model = DALLE(model_kwargs, is_reusable=True).to(model_kwargs["device"])

model.encoder.load_state_dict(
    torch.load("./will-dalle-files/mini/encoder.pt"), strict=False
)

model.decoder.load_state_dict(
    torch.load("./will-dalle-files/mini/decoder.pt"), strict=False
)

model.detokenizer.load_state_dict(torch.load("./will-dalle-files/detokenizer.pt"))

images = model(
    text="horses running on a beach",
    seed=-1,
    grid_size=3,
    grid=True,
    log2_supercondition_factor=8,
)

for idx, one in enumerate(images):
    one.save(f"test{idx}.png")
