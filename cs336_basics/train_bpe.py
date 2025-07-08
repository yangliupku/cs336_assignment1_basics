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
) -> list[bytes]:
    with open(input_path) as f:
        raw_text = f.read()
    special_tokens_escaped = [re.escape(t) for t in special_tokens]
    pattern = "|".join(special_tokens_escaped)
    parts = re.split(pattern, raw_text)
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    tokens = []
    for part in parts:
        tokens.extend([t.encode("utf-8") for t in regex.findall(PAT, part)])
    return tokens


def init_vocab(special_tokens: list[str]) -> dict[int, bytes]:
    vocab = {i: bytes([i]) for i in range(256)}  # Initialize with byte values
    curr_index = 256
    for token in special_tokens:
        if token not in vocab.values():
            vocab[curr_index] = token.encode("utf-8")
            curr_index += 1
    return vocab


def get_merge_pair(byte_tuple_list: list[tuple[bytes]]) -> tuple[bytes, bytes]:
    # count adjacent byte pairs
    c = Counter(byte_tuple_list)
    merge_counter = Counter()
    for byte_seq, seq_ct in c.items():
        if len(byte_seq) > 1:
            for i, j in pairwise(byte_seq):
                merge_counter[(i, j)] += seq_ct
    max_count = max(merge_counter.values())
    # break the tie and get most frequent pair
    tied_pairs = [pair for pair, count in merge_counter.items() if count == max_count]
    merge_byte_pair = sorted(tied_pairs, reverse=True)[0]
    return merge_byte_pair


def apply_merge_pair(byte_tuple_list: list[tuple[bytes]], merge_byte_pair: tuple[bytes, bytes]) -> list[tuple[bytes]]:
    merged_bytes_tuples = []
    for byte_seq in byte_tuple_list:
        new_byte_seq = []
        i = 0
        while i < len(byte_seq):
            if i < len(byte_seq) - 1 and (byte_seq[i], byte_seq[i + 1]) == merge_byte_pair:
                new_byte_seq.append(merge_byte_pair[0] + merge_byte_pair[1])
                i += 2
                i += 2
            else:
                new_byte_seq.append(byte_seq[i])
                i += 1
        merged_bytes_tuples.append(tuple(new_byte_seq))
    return merged_bytes_tuples


def train_bpe(
    input_path: str | os.PathLike,
    special_tokens: list[str] = ["<|endoftext|>"],
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    pre_tokenized_bytes = pretokenize(input_path, special_tokens)
    vocab = init_vocab(special_tokens)
    next_vocab_index = max(vocab.keys()) + 1
    merge_byte_pairs = []
    bytes_tuples = [tuple(bytes([i]) for i in list(word)) for word in pre_tokenized_bytes]
    print("input bytes", bytes_tuples)
    for merge_iteration in range(6):
        print(f"---------merge_iteration:{merge_iteration}--------------")
        merge_byte_pair = get_merge_pair(bytes_tuples)
        merge_byte_pairs.append(merge_byte_pair)
        print("merge_byte_pair", merge_byte_pair)
        vocab[next_vocab_index] = merge_byte_pair[0] + merge_byte_pair[1]
        next_vocab_index += 1
        bytes_tuples = apply_merge_pair(bytes_tuples, merge_byte_pair)
        print("bytes_tuples", bytes_tuples)
    return vocab, merge_byte_pairs


if __name__ == "__main__":
    input_file = DATA_PATH / "example.txt"
    # input_file = DATA_PATH / "TinyStoriesV2-GPT4-valid.txt"
    tokens = pretokenize(input_file)
    special_tokens = ["<|endoftext|>"]
    vocab, merges = train_bpe(input_file, special_tokens)
    print("vocab", vocab)
    print("merges", merges)
