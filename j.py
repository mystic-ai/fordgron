from gpt_k.model.gpt_j import GPTJForCausalLM
import time
import torch
from transformers import AutoTokenizer
from transformers import GPTJForCausalLM as TransformersGPTJ
from gpt_k.inference import stream_token_ids

tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B")

model_kwargs = {
    "embedding_dim": 4096,
    "num_attention_heads": 16,
    "mask": True,
    "dropout": 0,
    "forward_expansion": 4, # try 4
    "depth": 28,
    "seq_len": 2048,
    "vocab_len": 50400,
    "positional_encoding_implementation": "rotary_embedding",
    "rotary_pct": 0.25,
    "rotary_emb_base": 10000,
    "layernorm_eps": 1e-5,
    "eod_id": 0,
    "swap_batch_len_and_seq_len": False,
    "divide_qkv": True,
    "use_pa_ln": False,
    "device": 0
}

"""
To prevent allocation memory on CPU, we initialize on 'meta' and individually
port each module over to 'device' as we load each state dict.

:param checkpoint_path: Path to the checkpoint folder
:param use_cache: whether to use cache (i.e. for efficient generation)
:param device: device that you want the model to end up on
:return: model
"""

transformers = True

if transformers:
    gptj_from_hf = TransformersGPTJ.from_pretrained("EleutherAI/gpt-j-6B", revision="float16").half().to(0)
    input = "I met a traveller"
    now = time.time()
    model_input = tokenizer.encode(input, return_tensors="pt").to(0)
    test = gptj_from_hf.generate(model_input, max_length=400, min_length=400)
    print(time.time() - now)
    print(test)
else:
    weights = torch.load("gpt-j-weights.bin")
    model = GPTJForCausalLM(model_kwargs, device=0)
    model = model.half()
    model = model.to(device=model_kwargs["device"])
    model.transformer.token_embedding.load_state_dict({"weight": weights[f"transformer.wte.weight"]})
    for layer_i in range(model_kwargs["depth"]):
        model.transformer.h[layer_i].ln_1.load_state_dict({"weight": weights[f"transformer.h.{layer_i}.ln_1.weight"], "bias": weights[f"transformer.h.{layer_i}.ln_1.bias"]})
        model.transformer.h[layer_i].mlp.fc_in.load_state_dict({"weight": weights[f"transformer.h.{layer_i}.mlp.fc_in.weight"], "bias": weights[f"transformer.h.{layer_i}.mlp.fc_in.bias"]})
        model.transformer.h[layer_i].mlp.fc_out.load_state_dict({"weight": weights[f"transformer.h.{layer_i}.mlp.fc_out.weight"], "bias": weights[f"transformer.h.{layer_i}.mlp.fc_out.bias"]})
        model.transformer.h[layer_i].attn.q_proj.load_state_dict({"weight": weights[f"transformer.h.{layer_i}.attn.q_proj.weight"]})
        model.transformer.h[layer_i].attn.k_proj.load_state_dict({"weight": weights[f"transformer.h.{layer_i}.attn.k_proj.weight"]})
        model.transformer.h[layer_i].attn.v_proj.load_state_dict({"weight": weights[f"transformer.h.{layer_i}.attn.v_proj.weight"]})
        model.transformer.h[layer_i].attn.out_proj.load_state_dict({"weight": weights[f"transformer.h.{layer_i}.attn.out_proj.weight"]})
    model.transformer.ln_f.load_state_dict({"weight": weights[f"transformer.ln_f.weight"], "bias": weights[f"transformer.ln_f.bias"]})
    model.lm_head.load_state_dict({"weight": weights[f"lm_head.weight"], "bias": weights[f"lm_head.bias"]})
    del weights

input_data = "I met a traveller"

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
print(time.time() - now)


""" instantiation_times = []
half_times = []
to_empty_times = []

for _ in range(5):
    # Instantiate mode
    now = time.time()
    model = Transformer(model_kwargs, device="meta")
    instantiation_times.append(time.time() - now)
    now = time.time()
    model = model.half()
    half_times.append(time.time() - now)
    now = time.time()
    model = model.to(device=model_kwargs["device"])
    to_empty_times.append(time.time() - now)
    now = time.time()
    del model

print("instantiation")
print(sum(instantiation_times) / len(instantiation_times))
print("half times")
print(sum(half_times) / len(half_times))
print("to empty times")
print(sum(to_empty_times) / len(to_empty_times)) """