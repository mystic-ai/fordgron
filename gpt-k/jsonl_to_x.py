import argparse
import os
import re
import random

from pathlib import Path
from typing import List

import ftfy
from transformers import GPT2TokenizerFast
from tqdm import tqdm
import torch

# adapted from mesh-transformer-jax

def parse_args():
    parser = argparse.ArgumentParser(description="""
    Converts a text dataset into the training data format expected by the model.
    Adapted from the script create_tfrecords.py in the gpt-neo repo.
    - Your text dataset:
        - can be provided as .txt files, or as an archive (.tar.gz, .xz, jsonl.zst).
        - can be one file or multiple
            - using a single large file may use too much memory and crash - if this occurs, split the file up into a few files
        - the model's end-of-text separator is added between the contents of each file
        - if the string '<|endoftext|>' appears inside a file, it is treated as the model's end-of-text separator (not the actual string '<|endoftext|>')
            - this behavior can be disabled with --treat-eot-as-text
    This script creates a single .tfrecords file as output
        - Why: the model's data loader ignores "trailing" data (< 1 batch) at the end of a .tfrecords file
            - this causes data loss if you have many .tfrecords files
        - This is probably not appropriate for very large datasets
    """, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument(
        "input_path",
        type=str,
        help="Path to an input file, or a directory that contains the input files.",
    )
    parser.add_argument("name", type=str,
                        help="Name of output file will be {name}_{seqnum}.tfrecords, where seqnum is total sequence count")
    parser.add_argument("--output-dir", type=str, default="", help="Output directory (default: current directory)")

    cleaning_args = parser.add_argument_group('data cleaning arguments')

    cleaning_args.add_argument("--normalize-with-ftfy", action="store_true", help="Normalize text with ftfy")
    cleaning_args.add_argument("--normalize-with-wikitext-detokenize",
                               action="store_true", help="Use wikitext detokenizer")
    minu_help = "Exclude repetitive documents made up of < MIN_UNIQUE_TOKENS unique tokens. These can produce large gradients."
    minu_help += " Set <= 0 to disable. If enabled, 200 is a good default value. (Default: 0)"
    cleaning_args.add_argument("--min-unique-tokens", type=int, default=0,
                               help=minu_help)

    shuffle_pack_args = parser.add_argument_group('data shuffling/packing arguments')
    repack_ep_help = "Repeat the data num-repacks times, shuffled differently in each repetition. Recommended for multi-epoch training (set this to your intended number of epochs)."
    shuffle_pack_args.add_argument("--num-repacks",
                                   type=int, default=1,
                                   help=repack_ep_help
                                   )
    shuffle_pack_args.add_argument("--seed", type=int, default=10,
                                   help="random seed for shuffling data (default: 10)")
    shuffle_pack_args.add_argument("--preserve-data-order",
                                   default=False, action="store_true",
                                   help="Disables shuffling, so the input and output data have the same order.")

    misc_args = parser.add_argument_group('miscellaneous arguments')
    misc_args.add_argument("--verbose",
                           default=False, action="store_true",
                           help="Prints extra information, such as the text removed by --min-unique-tokens")

    args = parser.parse_args()

    # convert input_path to pathy
    args.input_path = Path(args.input_path)

    return args


def get_files(input_path):
    supported_types = ["jsonl.zst", ".txt", ".xz", ".tar.gz"]

    if input_path.is_dir():
        files = [list(Path(input_path).glob(f"*{type}")) for type in supported_types]
        files = [f for sublist in files for f in sublist]
        assert files, f"No files with supported types found in directory: {input_path}"
    elif input_path.is_file():
        assert any(
            str(input_path).endswith(type) for type in supported_types
        ), f"Input file type must be one of: {supported_types}"
        files = [input_path]
    else:
        raise FileNotFoundError(f"No such file or directory: {input_path=}")

    return [str(f) for f in files]


def wikitext_detokenizer(string):
    # contractions
    string = string.replace("s '", "s'")
    string = re.sub(r"/' [0-9]/", r"/'[0-9]/", string)
    # number separators
    string = string.replace(" @-@ ", "-")
    string = string.replace(" @,@ ", ",")
    string = string.replace(" @.@ ", ".")
    # punctuation
    string = string.replace(" : ", ": ")
    string = string.replace(" ; ", "; ")
    string = string.replace(" . ", ". ")
    string = string.replace(" ! ", "! ")
    string = string.replace(" ? ", "? ")
    string = string.replace(" , ", ", ")
    # double brackets
    string = re.sub(r"\(\s*([^\)]*?)\s*\)", r"(\1)", string)
    string = re.sub(r"\[\s*([^\]]*?)\s*\]", r"[\1]", string)
    string = re.sub(r"{\s*([^}]*?)\s*}", r"{\1}", string)
    string = re.sub(r"\"\s*([^\"]*?)\s*\"", r'"\1"', string)
    string = re.sub(r"'\s*([^']*?)\s*'", r"'\1'", string)
    # miscellaneous
    string = string.replace("= = = =", "====")
    string = string.replace("= = =", "===")
    string = string.replace("= =", "==")
    string = string.replace(" " + chr(176) + " ", chr(176))
    string = string.replace(" \n", "\n")
    string = string.replace("\n ", "\n")
    string = string.replace(" N ", " 1 ")
    string = string.replace(" 's", "'s")

    return string

def write_to_file(writer, data):
    """
    writes data to tfrecord file
    """
    feature = {
        "text": _int64_feature(data)
    }
    tf_example = tf.train.Example(features=tf.train.Features(feature=feature))
    writer.write(tf_example.SerializeToString())


def write_pt(sequences, file_path):
    torch.save(sequences, file_path)


def split_list(l, n):
    # splits list/string into n size chunks
    return [l[i:i + n] for i in range(0, len(l), n)]


def enforce_min_unique(seqs, min_unique_tokens, enc, verbose=False):
    for seq in tqdm(seqs, mininterval=1, smoothing=0, desc="enforce_min_unique_tokens"):
        if len(set(seq)) >= min_unique_tokens:
            yield seq
        elif verbose:
            text = enc.decode(seq)
            print(f"excluding with {len(set(seq))} unique tokens:\n\n{repr(text)}\n\n")


def eot_splitting_generator(string_iterable, tokenizer):
    """
    Given strings, splits them internally on <|endoftext|> and yields (generally more) strings
    """
    for doc in string_iterable:
        for d in doc.split(tokenizer.eos_token):
            if len(d) > 0:
                yield d


def prep_and_tokenize_generator(string_iterable, tokenizer, normalize_with_ftfy, normalize_with_wikitext_detokenize):
    for doc in string_iterable:
        if normalize_with_ftfy:
            doc = ftfy.fix_text(doc, normalization='NFKC')
        if normalize_with_wikitext_detokenize:
            doc = wikitext_detokenizer(doc)
        tokens = tokenizer.encode(doc) + [tokenizer.eos_token_id]
        yield tokens


def file_to_tokenized_docs_generator(file_path, tokenizer, args):
    with open(file_path, "r") as file:
      string_iterable = [file.read()]
      string_iterable = eot_splitting_generator(string_iterable, tokenizer)
      token_list_gen = prep_and_tokenize_generator(string_iterable,
                                                  tokenizer,
                                                  normalize_with_ftfy=args.normalize_with_ftfy,
                                                  normalize_with_wikitext_detokenize=args.normalize_with_wikitext_detokenize
                                                  )
      return token_list_gen


def read_files_to_tokenized_docs(files, args, encoder):
    docs = []

    if args.preserve_data_order:
        files = sorted(files)
    else:
        random.shuffle(files)

    for f in tqdm(files, mininterval=10, smoothing=0, desc="reading/tokenizing files"):
        docs.extend(file_to_tokenized_docs_generator(f, encoder, args))

    if not args.preserve_data_order:
        # shuffle at individual document level
        random.shuffle(docs)

    return docs


def arrays_to_sequences(token_list_iterable, sequence_length=2049):
    accumulating_sequence = []
    for l in token_list_iterable:
        accumulating_sequence.extend(l)

        if len(accumulating_sequence) > sequence_length:
            chunks = split_list(accumulating_sequence, sequence_length)
            yield from chunks[:-1]
            accumulating_sequence = chunks[-1]

    if len(accumulating_sequence) > 0:
        yield accumulating_sequence


def chunk_and_finalize(arrays, args, tokenizer):
    sequences = list(arrays_to_sequences(arrays))

    full_seqs, trailing_data = sequences[:-1], sequences[-1]

    if args.min_unique_tokens > 0:
        full_seqs = list(enforce_min_unique(full_seqs, args.min_unique_tokens, tokenizer, args.verbose))

    if not args.preserve_data_order:
        random.shuffle(full_seqs)

    return full_seqs, trailing_data


def create_pt_files(files, args):
    tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')

    random.seed(args.seed)

    # first
    sequences = []
    docs = read_files_to_tokenized_docs(files, args, tokenizer)
    full_seqs, trailing_data = chunk_and_finalize(docs, args, tokenizer)
    sequences.extend(full_seqs)

    # repacks
    for repeat_idx in range(1, args.num_repacks):
        if not args.preserve_data_order:
            random.shuffle(docs)
            full_seqs, trailing_data = chunk_and_finalize(docs, args, tokenizer)
        else:
            # if we're preserving data order, we can still "repack" by shifting everything
            # with the trailing data of the last epoch at the beginning
            seqs_with_prefix = [trailing_data] + full_seqs
            full_seqs, trailing_data = chunk_and_finalize(seqs_with_prefix, args, tokenizer)

        sequences.extend(full_seqs)

    # final
    print(f"dropped {len(trailing_data)} tokens of trailing data")

    total_sequence_len = len(sequences)

    new_file_path = os.path.join(args.output_dir, f"{args.name}_{total_sequence_len}.pt")
    write_pt(sequences, new_file_path)


if __name__ == "__main__":
    args = parse_args()

    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
    files = get_files(args.input_path)
    results = create_pt_files(files, args)
