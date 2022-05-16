#!/usr/bin/env python
# coding: utf-8

from gpt_k import model

import torch
from torch import nn
import torch.nn.functional as F
import torch.distributions as dist
import torch.distributed as distributed
from torch.nn.parallel import DistributedDataParallel as DDP
from sample import sample
from generate import generate
from get_dataloader import get_dataloader

import numpy as np

from argparse import ArgumentParser, Namespace
import random, tqdm, gzip
import os
from transformers import GPT2TokenizerFast

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # initialize me within process group
    distributed.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    # cleanup me within process group
    distributed.destroy_process_group()


import wandb

tokenizer = GPT2TokenizerFast.from_pretrained("EleutherAI/gpt-j-6B")

def print_rank_0(*message):
    """If distributed is initialized print only on rank 0."""
    if torch.distributed.is_initialized():
        if torch.distributed.get_rank() == 0:
            print(*message, flush=True)
    else:
        print(*message, flush=True)


def infer(rank, world_size, params):
    try:
        if rank == 0 and params.use_wandb == True:
                wandb.login(key="a45c4a8186b144484151295c474916a11a2d2bb5")
                wandb.init(project="gpt-k", entity="mystic-ai", config=vars(params))
        
        if params.seed < 0:
            seed = random.randint(0, 1000000)
        else:
            torch.manual_seed(params.seed)

        setup(rank, world_size)

        data_train = get_dataloader(params.batch_size)

        transformer = model.Transformer(
            params.embedding_dim,
            params.num_heads,
            params.mask,
            params.dropout,
            params.forward_expansion,
            params.depth,
            params.seq_len,
            params.vocab_len,
        ).to(rank)

        ddp_transformer = DDP(transformer, device_ids=[rank], output_device=rank)
        
        print("Number of parameters:")
        print("{:,}".format(sum(param.numel() for param in ddp_transformer.parameters() if param.requires_grad)))

        opt = torch.optim.Adam(lr=params.lr, params=ddp_transformer.parameters())
        sch = torch.optim.lr_scheduler.LambdaLR(
            opt, lambda i: min(i / (params.lr_warmup / params.batch_size), 1.0)
        )

        for i in tqdm.trange(params.num_batches, position=rank):
            opt.zero_grad()
            raw = next(iter(data_train))
            source = raw["text"][:,0:params.seq_len].to(torch.long)
            source_len = source.size(1)
            pad_len = params.seq_len - source_len
            if pad_len > 0:
                source = torch.cat([source, torch.zeros([1, pad_len], dtype=torch.long)], dim=1)
            data_test = raw["text"][0,:params.test_size]
            if rank != "cpu":
                source = source.to(rank)
            
            output = ddp_transformer(source) # [batch_len, seq_len, vocab_len]
            
            # target for the loss is the sequence 'shifted left'
            target = raw["text"][:,1:params.seq_len + pad_len + 1].to(torch.long) # [batch_len, seq_len]
            if rank != "cpu":
                target = target.to(rank)
            # negative log likelihood loss
            # with a mean reduction, the output's sum is divided by the output's length
            classes = output.transpose(1, 2) # [batch_len, vocab_len, seq_len] arrange sequences 'into columns' for comparison to target
            loss = F.nll_loss(classes, target, reduction="mean")
            loss.backward()
            if rank == 0 and params.use_wandb == True:
                wandb.log({"loss": loss})
            if params.gradient_clipping > 0.0:
                nn.utils.clip_grad_norm_(transformer.parameters(), params.gradient_clipping)
            opt.step()
            sch.step()

            if i != 0 and (i % params.test_every == 0 or i == params.num_batches - 1):
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

        cleanup()

    except Exception as err:
        print_rank_0(err)
        cleanup()
