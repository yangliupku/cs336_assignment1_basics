from collections import Counter
from itertools import pairwise
import pathlib
import pickle
import re
import os
import regex
from typing import BinaryIO
import multiprocessing as mp
import time
from tqdm import tqdm

DATA_PATH = (pathlib.Path(__file__).resolve().parent.parent) / "data"
OUTPUT_PATH = (pathlib.Path(__file__).resolve().parent.parent) / "bpe_output"
FIXUTRES_PATH = (pathlib.Path(__file__).resolve().parent.parent) / "tests" / "fixtures"


def find_chunk_boundaries(file: BinaryIO, desired_num_chunks: int, split_special_token: bytes) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))


def pretokenize_chunk(
    input_path: str | os.PathLike, chunk_start: int, chunk_end: int, special_tokens: list[str]
) -> dict[tuple[bytes], int]:
    with open(input_path, "rb") as f:
        f.seek(chunk_start)
        chunk = f.read(chunk_end - chunk_start).decode("utf-8", errors="ignore")
    special_tokens_escaped = [re.escape(t) for t in special_tokens]
    pattern = "|".join(special_tokens_escaped)
    parts = re.split(pattern, chunk)
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    tokens_dict = Counter()
    for part in parts:
        for t in regex.finditer(PAT, part):
            tokens_dict[t.group().encode("utf-8")] += 1
    # tokens_dict = {b'abc': 1, b'123': 2}
    tokens_tuple_dict = Counter()
    for k, v in tokens_dict.items():
        byte_tuple = tuple(bytes([i]) for i in list(k))
        # byte_tuple = (b'a', b'b', b'c')
        tokens_tuple_dict[byte_tuple] = v
    return tokens_tuple_dict


def pretokenize(
    input_path: str | os.PathLike,
    special_tokens: list[str] = ["<|endoftext|>"],
    num_processes: int = 8,
) -> dict[tuple[bytes], int]:
    start_time = time.time()
    chunk_args = []
    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")
    for star, end in pairwise(boundaries):
        chunk_args.append((input_path, star, end, special_tokens))

    with mp.Pool(processes=num_processes) as pool:
        results = pool.starmap(pretokenize_chunk, chunk_args)
    print(f"---> Pretokenization time: {time.time() - start_time:.2f}s")
    return sum(results, Counter())


def get_merge_pair(byte_tuple_dict: dict[tuple[bytes], int]) -> tuple[bytes, bytes]:
    # count adjacent byte pairs
    start_time = time.time()
    merge_counter = Counter()
    for byte_seq, seq_ct in byte_tuple_dict.items():
        if len(byte_seq) > 1:
            for i, j in pairwise(byte_seq):
                merge_counter[(i, j)] += seq_ct
    max_count = max(merge_counter.values())
    # break the tie and get most frequent pair
    tied_pairs = [pair for pair, count in merge_counter.items() if count == max_count]
    merge_byte_pair = max(tied_pairs)
    tqdm.write(f"---> get_merge_pair time: {time.time() - start_time:.2f}s")
    return merge_byte_pair


def apply_merge_pair(
    byte_tuple_dict: dict[tuple[bytes], int], merge_byte_pair: tuple[bytes, bytes]
) -> dict[tuple[bytes], int]:
    start_time = time.time()
    merged_bytes_tuple_dict = Counter()
    for byte_tuple, count in byte_tuple_dict.items():
        new_byte_seq = []
        i = 0
        while i < len(byte_tuple):
            if i < len(byte_tuple) - 1 and (byte_tuple[i], byte_tuple[i + 1]) == merge_byte_pair:
                new_byte_seq.append(merge_byte_pair[0] + merge_byte_pair[1])
                i += 2
            else:
                new_byte_seq.append(byte_tuple[i])
                i += 1
        merged_bytes_tuple_dict[tuple(new_byte_seq)] += count
    tqdm.write(f"---> apply_merge_pair time: {time.time() - start_time:.2f}s")
    return merged_bytes_tuple_dict


def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str] = ["<|endoftext|>"],
    num_processes: int = 8,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    bytes_tuple_dict = pretokenize(input_path, special_tokens, num_processes)
    vocab_list = [t.encode("utf-8") for t in special_tokens]
    for i in range(256):
        if bytes([i]) not in vocab_list:
            vocab_list.append(bytes([i]))

    merge_byte_pairs = []
    with tqdm(total=vocab_size, initial=len(vocab_list), desc="Building vocab") as pbar:
        while len(vocab_list) < vocab_size:
            merge_byte_pair = get_merge_pair(bytes_tuple_dict)
            merge_byte_pairs.append(merge_byte_pair)
            vocab_list.append(merge_byte_pair[0] + merge_byte_pair[1])
            bytes_tuple_dict = apply_merge_pair(bytes_tuple_dict, merge_byte_pair)
            pbar.update(1)
    vocab = {i: v for i, v in enumerate(vocab_list)}
    return vocab, merge_byte_pairs


def train_bpe_tinystories():
    start_time = time.time()
    input_file = DATA_PATH / "TinyStoriesV2-GPT4-train.txt"
    special_tokens = ["<|endoftext|>"]
    vocab, merges = train_bpe(input_file, 10000, special_tokens)
    with open(OUTPUT_PATH / "tinystories.pkl", "wb") as f:
        pickle.dump({"vocab": vocab, "merges": merges}, f)
    print(f"-------> Training elapsed time: {time.time() - start_time:.2f}s")


if __name__ == "__main__":
    # input_file = FIXUTRES_PATH / "tinystories_sample_5M.txt"
    # special_tokens = ["<|endoftext|>"]
    # vocab, merges = train_bpe(input_file, 1000, special_tokens)
    # with open(OUTPUT_PATH / "test.pkl", "wb") as f:
    #     pickle.dump({"vocab": vocab, "merges": merges}, f)

    train_bpe_tinystories()
