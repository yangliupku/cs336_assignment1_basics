from collections import Counter
from itertools import pairwise
import pathlib
import re
import os
import regex

DATA_PATH = (pathlib.Path(__file__).resolve().parent.parent) / "data"


def pretokenize(
    input_path: str | os.PathLike,
    special_tokens: list[str] = ["<|endoftext|>"],
) -> dict[tuple[bytes], int]:
    with open(input_path) as f:
        raw_text = f.read()
    special_tokens_escaped = [re.escape(t) for t in special_tokens]
    pattern = "|".join(special_tokens_escaped)
    parts = re.split(pattern, raw_text)
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    tokens_dict = Counter()
    for part in parts:
        for t in regex.findall(PAT, part):
            tokens_dict[t.encode("utf-8")] += 1
    # tokens_dict = {b'abc': 1, b'123': 2}
    tokens_tuple_dict = Counter()
    for k, v in tokens_dict.items():
        byte_tuple = tuple(bytes([i]) for i in list(k))
        # byte_tuple = (b'a', b'b', b'c')
        tokens_tuple_dict[byte_tuple] = v
    return tokens_tuple_dict


def init_vocab(special_tokens: list[str]) -> dict[int, bytes]:
    vocab = {i: bytes([i]) for i in range(256)}  # Initialize with byte values
    curr_index = 256
    for token in special_tokens:
        if token not in vocab.values():
            vocab[curr_index] = token.encode("utf-8")
            curr_index += 1
    return vocab


def get_merge_pair(byte_tuple_dict: dict[tuple[bytes], int]) -> tuple[bytes, bytes]:
    # count adjacent byte pairs
    merge_counter = Counter()
    for byte_seq, seq_ct in byte_tuple_dict.items():
        if len(byte_seq) > 1:
            for i, j in pairwise(byte_seq):
                merge_counter[(i, j)] += seq_ct
    max_count = max(merge_counter.values())
    # break the tie and get most frequent pair
    tied_pairs = [pair for pair, count in merge_counter.items() if count == max_count]
    merge_byte_pair = sorted(tied_pairs, reverse=True)[0]
    return merge_byte_pair


def apply_merge_pair(
    byte_tuple_dict: dict[tuple[bytes], int], merge_byte_pair: tuple[bytes, bytes]
) -> dict[tuple[bytes], int]:
    merged_bytes_tuple_dict = Counter()
    for byte_tuple, count in byte_tuple_dict.items():
        new_byte_seq = []
        i = 0
        while i < len(byte_tuple):
            if i < len(byte_tuple) - 1 and (byte_tuple[i], byte_tuple[i + 1]) == merge_byte_pair:
                new_byte_seq.append(merge_byte_pair[0] + merge_byte_pair[1])
                i += 2
                i += 2
            else:
                new_byte_seq.append(byte_tuple[i])
                i += 1
        merged_bytes_tuple_dict[tuple(new_byte_seq)] += count
    return merged_bytes_tuple_dict


def train_bpe(
    input_path: str | os.PathLike,
    special_tokens: list[str] = ["<|endoftext|>"],
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    bytes_tuple_dict = pretokenize(input_path, special_tokens)
    vocab = init_vocab(special_tokens)
    merge_byte_pairs = []
    next_vocab_index = max(vocab.keys()) + 1
    for merge_iteration in range(6):
        print(f"---------merge_iteration:{merge_iteration}--------------")
        merge_byte_pair = get_merge_pair(bytes_tuple_dict)
        merge_byte_pairs.append(merge_byte_pair)
        print("merge_byte_pair", merge_byte_pair)
        vocab[next_vocab_index] = merge_byte_pair[0] + merge_byte_pair[1]
        next_vocab_index += 1
        bytes_tuple_dict = apply_merge_pair(bytes_tuple_dict, merge_byte_pair)
    return vocab, merge_byte_pairs


if __name__ == "__main__":
    input_file = DATA_PATH / "example.txt"
    input_file = DATA_PATH / "TinyStoriesV2-GPT4-valid.txt"
    # tokens = pretokenize(input_file)
    special_tokens = ["<|endoftext|>"]
    vocab, merges = train_bpe(input_file, special_tokens)
    print("vocab", vocab)
    print("merges", merges)
