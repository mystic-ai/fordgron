import transformers
from transformers import AutoModelForCausalLM
import time
import torch

now = time.time()


class no_init:
    def __init__(self, modules=None, use_hf_no_init=True):
        if modules is None:
            self.modules = [
                torch.nn.Linear,
                torch.nn.Embedding,
                torch.nn.LayerNorm,
            ]
        self.original = {}
        self.use_hf_no_init = use_hf_no_init

    def __enter__(self):
        if self.use_hf_no_init:
            transformers.modeling_utils._init_weights = False
        for mod in self.modules:
            self.original[mod] = getattr(mod, "reset_parameters", None)
            mod.reset_parameters = lambda x: x

    def __exit__(self, type, value, traceback):
        if self.use_hf_no_init:
            transformers.modeling_utils._init_weights = True
        for mod in self.modules:
            setattr(mod, "reset_parameters", self.original[mod])


with no_init():
    model = (
        AutoModelForCausalLM.from_pretrained(
            "EleutherAI/gpt-j-6B",
            revision="float16",
            torch_dtype=torch.float16,
        )
        .half()
        .to(0)
    )
print(time.time() - now)
