from k import model

import torch
from torch import nn
import torch.nn.functional as F
import torch.distributions as dist

import numpy as np

from argparse import ArgumentParser
import random, tqdm, gzip
import os


def sample(lnprobs, temperature=1.0):
    if temperature == 0.0:
        return lnprobs.argmax()

    p = F.softmax(lnprobs / temperature, dim=0)
    cd = dist.Categorical(p)
    return cd.sample()


def divide_dataset(path, max_size=10e8):
    data_size = min(os.path.getsize(path), max_size)
    training_size = int(0.9 * data_size)
    validation_size = int(0.05 * data_size)
    test_size = int(0.05 * data_size)
    with gzip.open(path) if path.endswith(".gz") else open(path) as file:
        X = np.fromstring(
            file.read(training_size + validation_size + test_size), dtype=np.uint8
        )
        trX, vaX, teX = np.split(X, [training_size, training_size + validation_size])
        return torch.from_numpy(trX), torch.from_numpy(vaX), torch.from_numpy(teX)


def sample_batch(data, seq_len, batch_size):
    starts = torch.randint(size=(batch_size,), low=0, high=data.size(0) - seq_len - 1)
    seqs_inputs = [data[start : start + seq_len] for start in starts]
    seqs_target = [data[start + 1 : start + seq_len + 1] for start in starts]
    inputs = torch.cat([s[None, :] for s in seqs_inputs], dim=0).to(torch.long)
    target = torch.cat([s[None, :] for s in seqs_target], dim=0).to(torch.long)
    return inputs, target


def sample_sequence(
    transformer, seed, max_context, length=600, temperature=0.5, verbose=False
):
    sequence = seed.detach().clone()

    if verbose:
        print("Prompt:")
        for c in seed:
            print(str(chr(c)), end="", flush=True)
        print()

    print("Generation:")
    for _ in range(length):
        input = sequence[-max_context:]
        output = transformer(input[None, :])
        c = sample(output[0, -1, :], temperature)

        if verbose:
            print(str(chr(max(32, c))), end="", flush=True)

        sequence = torch.cat([sequence, c[None]], dim=0)
    return sequence


def infer(arg):
    if arg.seed < 0:
        seed = random.randint(0, 1000000)
    else:
        torch.manual_seed(arg.seed)

    data_train, data_val, data_test = divide_dataset(arg.data)
    data_train, data_test = (
        (torch.cat([data_train, data_val], dim=0), data_test)
        if arg.final
        else (data_train, data_val)
    )

    transformer = model.Transformer(
        arg.embedding_dim,
        arg.num_heads,
        arg.mask,
        arg.dropout,
        arg.forward_expansion,
        arg.depth,
        arg.seq_len,
        arg.vocab_len,
        arg.device,
    ).to(arg.device)

    opt = torch.optim.Adam(lr=arg.lr, params=transformer.parameters())
    sch = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda i: min(i / (arg.lr_warmup / arg.batch_size), 1.0)
    )

    instances_seen = 0
    for i in tqdm.trange(arg.num_batches):
        opt.zero_grad()
        source, target = sample_batch(data_train, arg.seq_len, arg.batch_size)
        instances_seen += source.size(0)
        if torch.cuda.is_available():
            source, target = source.cuda(), target.cuda()
        output = transformer(source)
        loss = F.nll_loss(output.transpose(2, 1), target, reduction="mean")
        loss.backward()
        if arg.gradient_clipping > 0.0:
            nn.utils.clip_grad_norm_(transformer.parameters(), arg.gradient_clipping)
        opt.step()
        sch.step()

        if i != 0 and (i % arg.test_every == 0 or i == arg.num_batches - 1):
            with torch.no_grad():
                seedfr = random.randint(0, data_test.size(0) - arg.seq_len)
                seed = data_test[seedfr : seedfr + arg.seq_len].to(torch.long)
                if torch.cuda.is_available():
                    seed = seed.cuda()

                sample_sequence(
                    transformer,
                    seed,
                    max_context=arg.seq_len,
                    verbose=True,
                    length=arg.sample_length,
                )


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument(
        "--num-batches",
        dest="num_batches",
        help="Number of batches to train on. Each batch contains randomly sampled subsequences of the data."
        "Default is set to a very large value so you can keep running until the output looks good. ",
        default=1_000_000,
        type=int,
    )

    parser.add_argument(
        "--batch-size",
        dest="batch_size",
        help="The batch size.",
        default=32,
        type=int,
    )

    parser.add_argument(
        "--data",
        dest="data",
        help="Data file. Will be read as a string of 8-bit characters.",
        default=None,
    )

    parser.add_argument(
        "--embedding",
        dest="embedding_dim",
        help="Size of the character embeddings.",
        default=128,
        type=int,
    )

    parser.add_argument(
        "--heads",
        dest="num_heads",
        help="Number of attention heads.",
        default=8,
        type=int,
    )

    parser.add_argument(
        "--mask",
        dest="mask",
        help="Whether to mask or not.",
        action="store_true",
    )

    parser.add_argument(
        "--dropout",
        dest="dropout",
        help="Dropout.",
        default=0.8,
        type=float,
    )

    parser.add_argument(
        "--forward_expansion",
        dest="forward_expansion",
        help="Forward expansion.",
        default=4,
        type=int,
    )

    parser.add_argument(
        "--depth",
        dest="depth",
        help="Depth of the network (nr. of transformer blocks)",
        default=12,
        type=int,
    )

    parser.add_argument(
        "--seq-len",
        dest="seq_len",
        help="Length of the sequences extracted from the corpus (and the context used during inference).",
        default=256,
        type=int,
    )

    parser.add_argument(
        "--vocab-len",
        dest="vocab_len",
        help="Number of tokens in the vocabulary.",
        default=256,
        type=int,
    )

    parser.add_argument(
        "--device",
        dest="device",
        help="What device.",
        default="cuda",
        type=str,
    )

    parser.add_argument(
        "--random-seed",
        dest="seed",
        help="RNG seed. Negative for random",
        default=-1,
        type=int,
    )

    parser.add_argument(
        "--learn-rate",
        dest="lr",
        help="Learning rate",
        default=0.0001,
        type=float,
    )

    parser.add_argument(
        "--final",
        dest="final",
        help="Whether to run on the real test set (if not included, the validation set is used).",
        action="store_true",
    )

    parser.add_argument(
        "--test-every",
        dest="test_every",
        help="How many batches between tests.",
        default=1500,
        type=int,
    )

    parser.add_argument(
        "--test-subset",
        dest="test_subset",
        help="A subset for the validation tests.",
        default=100000,
        type=int,
    )

    parser.add_argument(
        "--gradient-clipping",
        dest="gradient_clipping",
        help="Gradient clipping.",
        default=1.0,
        type=float,
    )

    parser.add_argument(
        "--lr-warmup",
        dest="lr_warmup",
        help="Learning rate warmup.",
        default=5000,
        type=int,
    )

    parser.add_argument(
        "--sample-length",
        dest="sample_length",
        help="Number of characters to sample.",
        default=600,
        type=int,
    )
    options = parser.parse_args()
    print(options)
    infer(options)
