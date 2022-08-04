from fordgron.model.gpt_j import GPTJForCausalLM
import time
import torch
import wandb
from rich.progress import track
import torch.nn.functional as F
from torch import nn

torch.autograd.set_grad_enabled(True)

test = torch.load("../medium/medium_2_4_1224.pt")

model_kwargs = {
    "embedding_dim": 4096,
    "num_attention_heads": 16,
    "mask": True,
    "dropout": 0,
    "forward_expansion": 4,  # try 4
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
    "device": 3,
}

model = GPTJForCausalLM(model_kwargs, device="meta")
model = model.half()
model = model.to_empty(device=model_kwargs["device"])

weights = torch.load("gpt-j-weights.bin")  # 7 seconds
model.transformer.token_embedding.load_state_dict(
    {"weight": weights[f"transformer.wte.weight"]}
)
for layer_i in range(model_kwargs["depth"]):
    model.transformer.h[layer_i].ln_1.load_state_dict(
        {
            "weight": weights[f"transformer.h.{layer_i}.ln_1.weight"],
            "bias": weights[f"transformer.h.{layer_i}.ln_1.bias"],
        }
    )
    model.transformer.h[layer_i].mlp.fc_in.load_state_dict(
        {
            "weight": weights[f"transformer.h.{layer_i}.mlp.fc_in.weight"],
            "bias": weights[f"transformer.h.{layer_i}.mlp.fc_in.bias"],
        }
    )
    model.transformer.h[layer_i].mlp.fc_out.load_state_dict(
        {
            "weight": weights[f"transformer.h.{layer_i}.mlp.fc_out.weight"],
            "bias": weights[f"transformer.h.{layer_i}.mlp.fc_out.bias"],
        }
    )
    model.transformer.h[layer_i].attn.q_proj.load_state_dict(
        {"weight": weights[f"transformer.h.{layer_i}.attn.q_proj.weight"]}
    )
    model.transformer.h[layer_i].attn.k_proj.load_state_dict(
        {"weight": weights[f"transformer.h.{layer_i}.attn.k_proj.weight"]}
    )
    model.transformer.h[layer_i].attn.v_proj.load_state_dict(
        {"weight": weights[f"transformer.h.{layer_i}.attn.v_proj.weight"]}
    )
    model.transformer.h[layer_i].attn.out_proj.load_state_dict(
        {"weight": weights[f"transformer.h.{layer_i}.attn.out_proj.weight"]}
    )
model.transformer.ln_f.load_state_dict(
    {
        "weight": weights[f"transformer.ln_f.weight"],
        "bias": weights[f"transformer.ln_f.bias"],
    }
)
model.lm_head.load_state_dict(
    {"weight": weights[f"lm_head.weight"], "bias": weights[f"lm_head.bias"]}
)
del weights

training_kwargs = {"use_wandb": False, "lr": 0.1, "lr_warmup": 0.01, "batch_size": 2}

if training_kwargs["use_wandb"]:
    wandb.init(project="fordgron", entity="mystic-ai", config=training_kwargs)

opt = torch.optim.Adam(lr=training_kwargs["lr"], params=model.parameters())
sch = torch.optim.lr_scheduler.LambdaLR(
    opt,
    lambda i: min(
        i / (training_kwargs["lr_warmup"] / training_kwargs["batch_size"]), 1.0
    ),
)

model.train()

for sequence in track(test):
    opt.zero_grad()
    raw = test[0]
    source = raw[: model_kwargs["seq_len"]].to(torch.int32)
    source_len = source.size(0)
    pad_len = model_kwargs["seq_len"] - source_len
    if pad_len > 0:
        source = torch.cat(
            [source, torch.zeros([1, pad_len], dtype=torch.int32)], dim=1
        )
    # data_test = raw["text"][0, : params.test_size]
    source = source.to(model_kwargs["device"]).unsqueeze(0)

    output, _ = model(source)  # [batch_len, seq_len, vocab_len]

    # target for the loss is the sequence 'shifted left'
    target = raw[1 : model_kwargs["seq_len"] + pad_len + 1]  # [batch_len, seq_len]
    target = target.to(model_kwargs["device"]).unsqueeze(0)
    # negative log likelihood loss
    # with a mean reduction, the output's sum is divided by the output's length

    loss = F.cross_entropy(
        output.flatten(0, -2),
        target.flatten(),
        reduction="mean",
    )

    loss.backward()
    if training_kwargs["use_wandb"]:
        wandb.log({"loss": loss})
    """         if params.gradient_clipping > 0.0:
        nn.utils.clip_grad_norm_(transformer.parameters(), params.gradient_clipping) """
    opt.step()
    sch.step()

"""         if i != 0 and (i % params.test_every == 0 or i == params.num_batches - 1):
        with torch.no_grad():
            prompt = data_test.to(torch.long)
            if params.device != "cpu" and torch.cuda.is_available():
                prompt = prompt.to(rank)

            generate(
                ddp_transformer,
                tokenizer,
                prompt,
                max_context=params.seq_len,
                verbose=True,
                length=params.sample_length,
            )
"""
