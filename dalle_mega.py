from fordgron import DALLE
import torch
from os.path import exists
import requests

required_files = [
    (
        "https://huggingface.co/kuprel/min-dalle/resolve/main/encoder.pt",
        "./will-dalle-files/mega/encoder.pt",
    ),
    (
        "https://huggingface.co/kuprel/min-dalle/resolve/main/decoder.pt",
        "./will-dalle-files/mega/decoder.pt",
    ),
    (
        "https://huggingface.co/kuprel/min-dalle/resolve/main/vocab.json",
        "./will-dalle-files/mega/vocab.json",
    ),
    (
        "https://huggingface.co/kuprel/min-dalle/resolve/main/merges.json",
        "./will-dalle-files/mega/merges.txt",
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
    "depth": 24,
    "num_attention_heads": 32,
    "embedding_dim": 2048,
    "glu_embedding_dim": 4096,
    "text_vocab_len": 50272,
    "image_vocab_len": 16415,
    "vocab_path": "./will-dalle-files/mega/vocab.json",
    "merges_path": "./will-dalle-files/mega/merges.txt",
    "device": 0,
    "dtype": torch.float32,
}

model = DALLE(model_kwargs).to(model_kwargs["device"])

model.encoder.load_state_dict(
    torch.load("./will-dalle-files/mega/encoder.pt"), strict=False
)

model.decoder.load_state_dict(
    torch.load("./will-dalle-files/mega/decoder.pt"), strict=False
)

model.detokenizer.load_state_dict(torch.load("./will-dalle-files/detokenizer.pt"))


images = model(
    text="horses running on a beach",
    seed=-1,
    grid_size=3,
    grid=True,
)

for i, image in enumerate(images):
    image.save(f"image_{i}.png")
