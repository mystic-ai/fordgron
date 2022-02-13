#!/usr/bin/env python
# coding: utf-8

import torch.multiprocessing as mp
from ddp_train import infer
from argparse import Namespace
import wandb

num_batches = 90000
batch_size = 32
embedding_dim = 256
num_heads = 8
mask = True
dropout = 0.05
forward_expansion = 4
depth = 8
seq_len = 50
vocab_len = 52015
device = "cuda"
seed = -1
lr = 0.001
final = False
test_every = 200
gradient_clipping = 1
lr_warmup = 5000
sample_length = 600
test_size = 25

params = Namespace(
    num_batches=num_batches,
    batch_size=batch_size,
    embedding_dim=embedding_dim,
    num_heads=num_heads,
    mask=mask,
    dropout=dropout,
    forward_expansion=forward_expansion,
    depth=depth,
    seq_len=seq_len,
    vocab_len=vocab_len,
    device=device,
    seed=seed,
    lr=lr,
    final=final,
    test_every=test_every,
    gradient_clipping=gradient_clipping,
    lr_warmup=lr_warmup,
    sample_length=sample_length,
    test_size = test_size
)

world_size = 1

if __name__ == "__main__":
    mp.spawn(infer, args=(world_size, params), nprocs=world_size, join=True)