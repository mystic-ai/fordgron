#!/usr/bin/env python
# coding: utf-8

import torch.multiprocessing as mp
from infer import infer
from argparse import Namespace
import wandb

parameters = {
    "num_batches": 10000,
    "batch_size": 80,
    "embedding_dim": 512,
    "num_heads": 8,
    "mask": True,
    "dropout": 0.05,
    "forward_expansion": 2,
    "depth": 8,
    "seq_len": 100,
    "vocab_len": 52015,
    "device": "cuda",
    "seed": -1,
    "lr": 0.0001,
    "final": False,
    "test_every": 500,
    "gradient_clipping": 1,
    "lr_warmup": 5000,
    "sample_length": 600,
    "test_size": 25,
    "use_wandb": True
}

params = Namespace(**parameters)

world_size = 1

if __name__ == "__main__":
    mp.spawn(infer, args=(world_size, params), nprocs=world_size, join=True)