from gpt_k.model.dall_e import MinDalle
import time
import torch
from transformers import AutoTokenizer
from transformers import GPTJForCausalLM as TransformersGPTJ
from gpt_k.inference import stream_token_ids
from os.path import exists
import requests

tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B")

model_kwargs = {
    "depth": 24,
    "num_attention_heads": 32,
    "embedding_dim": 2048,
    "glu_embedding_dim": 4096,
    "text_vocab_len": 50272,
    "image_vocab_len": 16415,
    "vocab_path": "./will-dalle-files/mega/vocab.json",
    "merges_path": "./will-dalle-files/mega/merges.txt",
    "device": 0
}

required_files = [
    ("https://huggingface.co/kuprel/min-dalle/resolve/main/encoder.pt", "./will-dalle-files/mega/encoder.pt"),
    ("https://huggingface.co/kuprel/min-dalle/resolve/main/decoder.pt", "./will-dalle-files/mega/decoder.pt"),
    ("https://huggingface.co/kuprel/min-dalle/resolve/main/vocab.json", "./will-dalle-files/mega/vocab.json"),
    ("https://huggingface.co/kuprel/min-dalle/resolve/main/merges.json", "./will-dalle-files/mega/merges.txt"),
    ("https://huggingface.co/kuprel/min-dalle/resolve/main/detoker.pt", "./will-dalle-files/detokenizer.pt"),
]

def download_file(url: str, save_path: str):
    print(f"downloading {url}")
    download = requests.get(url)
    with open(save_path, 'wb') as f: f.write(download.content)

for file in required_files:
    if not exists(file[1]):
        download_file(file[0], file[1])

model = MinDalle(model_kwargs, is_reusable=True).to(model_kwargs["device"])
model.encoder.load_state_dict(torch.load("./will-dalle-files/mega/encoder.pt"), strict=False)
model.decoder.load_state_dict(torch.load("./will-dalle-files/mega/decoder.pt"), strict=False)
model.detokenizer.load_state_dict(torch.load("./will-dalle-files/detokenizer.pt"))

images = model(
    text='desk with rubix cube, grapes, satsuma, and a man working on a laptop', 
    seed=-1,
    grid_size=3,
)

for index, image in enumerate(images):
    image.save(f"test{index}.jpg")

""" input_data = "I met a traveller"

default_inference_kwargs = {
    "streaming": False,
    "generation_length": 400,
    "temperature": 1.0,
    "top_k": 0,
    "top_p": 0,
    "use_cache": True
}

inference_kwargs = {
    **default_inference_kwargs,
}

assert any(
    [isinstance(input_data, str), isinstance(input_data, list)]
), "Input should be a string, or a list of strings"
if isinstance(input_data, str):
    input_data = [input_data]

# streaming should not be allowed with a batch

num_batches = len(input_data)
input_pos = 0

# generate completions
generated_texts = []
terminate_runs = 0
if input_pos == num_batches:
    terminate_runs = 1
else:
    raw_text = input_data[input_pos]
    input_pos += 1

    if raw_text == "":
        context_tokens = [eos_token_id]
    else:
        context_tokens = tokenizer.encode(raw_text)
    context_length = len(context_tokens)

sentences = ["", "", "", "", ""]

now = time.time()
for (
    generated_tokens
) in stream_token_ids(
    model,
    inference_kwargs,
    input_tokens=[context_tokens], # wrapped in an outer list to emulate batch_len 1
    model_seq_len=2048
):
    try:
        for index, token in enumerate(generated_tokens.tolist()):
            sentences[index] = sentences[index] + tokenizer.decode(token)
        if True:
            print(sentences)
    except Exception as e:
        print(e)
        print("stream tokens failed")
print(time.time() - now) """