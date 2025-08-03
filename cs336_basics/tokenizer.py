from collections import Counter, defaultdict
from collections.abc import Iterable
from itertools import pairwise
import pathlib
import pickle
import re
import os
import regex
from typing import BinaryIO
import multiprocessing as mp
from tqdm import tqdm
from copy import deepcopy


DATA_PATH = (pathlib.Path(__file__).resolve().parent.parent) / "data"
OUTPUT_PATH = (pathlib.Path(__file__).resolve().parent.parent) / "bpe_output"
FIXUTRES_PATH = (pathlib.Path(__file__).resolve().parent.parent) / "tests" / "fixtures"
SHOW_PROGRESS = False


def find_chunk_boundaries(
    file: BinaryIO, desired_num_chunks: int, split_special_token: bytes
) -> list[int]:
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
    chunk_args = []
    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")
    for star, end in pairwise(boundaries):
        chunk_args.append((input_path, star, end, special_tokens))

    with mp.Pool(processes=num_processes) as pool:
        results = pool.starmap(pretokenize_chunk, chunk_args)
    return sum(results, Counter())


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
    merge_byte_pair = max(tied_pairs)
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
            else:
                new_byte_seq.append(byte_tuple[i])
                i += 1
        merged_bytes_tuple_dict[tuple(new_byte_seq)] += count
    return merged_bytes_tuple_dict


def init_token_pairs(
    byte_tuple_dict: dict[tuple[bytes], int],
) -> tuple[Counter[tuple[bytes, bytes], int], dict[tuple[bytes, bytes], set]]:
    token_pair_counter: Counter[tuple[bytes, bytes], int] = Counter()
    token_pair_src: dict[tuple[bytes, bytes], set] = defaultdict(set)
    for byte_seq, seq_ct in byte_tuple_dict.items():
        if len(byte_seq) > 1:
            for i, j in pairwise(byte_seq):
                token_pair_counter[(i, j)] += seq_ct
                token_pair_src[(i, j)].add(byte_seq)
    return token_pair_counter, token_pair_src


def get_merge_pair_with_cache(
    token_pair_counter: dict[tuple[bytes, bytes], int],
):
    max_count = max(token_pair_counter.values())
    # break the tie and get most frequent pair
    tied_pairs = [pair for pair, count in token_pair_counter.items() if count == max_count]
    merge_byte_pair = max(tied_pairs)
    return merge_byte_pair


def update_token_pairs(
    byte_tuple_dict: dict[tuple[bytes], int],
    token_pair_counter: dict[tuple[bytes, bytes], int],
    token_pair_src: dict[tuple[bytes, bytes], set],
    merge_byte_pair: tuple[bytes, bytes],
):
    token_tuples_to_update = deepcopy(token_pair_src[merge_byte_pair])
    # subtract all token pairs counts
    for byte_tuple in token_tuples_to_update:
        seq_ct = byte_tuple_dict[byte_tuple]
        for i, j in pairwise(byte_tuple):
            token_pair_counter[(i, j)] -= seq_ct
            token_pair_src[(i, j)].discard(byte_tuple)

    # merge the byte paire and update the pair counters
    for byte_tuple in token_tuples_to_update:
        seq_ct = byte_tuple_dict[byte_tuple]
        new_byte_seq = []
        i = 0
        while i < len(byte_tuple):
            if i < len(byte_tuple) - 1 and (byte_tuple[i], byte_tuple[i + 1]) == merge_byte_pair:
                new_byte_seq.append(merge_byte_pair[0] + merge_byte_pair[1])
                i += 2
            else:
                new_byte_seq.append(byte_tuple[i])
                i += 1
        byte_tuple_dict[tuple(new_byte_seq)] += seq_ct
        for i, j in pairwise(new_byte_seq):
            token_pair_counter[(i, j)] += seq_ct
            token_pair_src[(i, j)].add(tuple(new_byte_seq))

        byte_tuple_dict.pop(byte_tuple)

    token_pair_counter.pop(merge_byte_pair)
    token_pair_src.pop(merge_byte_pair)

    return


def train_bpe_old(
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
    with tqdm(
        total=vocab_size, initial=len(vocab_list), desc="Building vocab", disable=not SHOW_PROGRESS
    ) as pbar:
        while len(vocab_list) < vocab_size:
            merge_byte_pair = get_merge_pair(bytes_tuple_dict)
            merge_byte_pairs.append(merge_byte_pair)
            vocab_list.append(merge_byte_pair[0] + merge_byte_pair[1])
            bytes_tuple_dict = apply_merge_pair(bytes_tuple_dict, merge_byte_pair)
            pbar.update(1)
    vocab = {i: v for i, v in enumerate(vocab_list)}
    return vocab, merge_byte_pairs


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
    token_pair_counter, token_pair_src = init_token_pairs(bytes_tuple_dict)

    with tqdm(
        total=vocab_size, initial=len(vocab_list), desc="Building vocab", disable=not SHOW_PROGRESS
    ) as pbar:
        while len(vocab_list) < vocab_size:
            merge_byte_pair = get_merge_pair_with_cache(token_pair_counter)
            vocab_list.append(merge_byte_pair[0] + merge_byte_pair[1])
            merge_byte_pairs.append(merge_byte_pair)
            update_token_pairs(
                bytes_tuple_dict, token_pair_counter, token_pair_src, merge_byte_pair
            )
            pbar.update(1)
    vocab = {i: v for i, v in enumerate(vocab_list)}
    return vocab, merge_byte_pairs


def train_bpe_tinystories():
    input_file = DATA_PATH / "TinyStoriesV2-GPT4-train.txt"
    special_tokens = ["<|endoftext|>"]
    vocab, merges = train_bpe(input_file, 10000, special_tokens)
    with open(OUTPUT_PATH / "tinystories.pkl", "wb") as f:
        pickle.dump({"vocab": vocab, "merges": merges}, f)


def train_bpe_owt():
    input_file = DATA_PATH / "owt_train.txt"
    special_tokens = ["<|endoftext|>"]
    vocab, merges = train_bpe(input_file, 32000, special_tokens)
    with open(OUTPUT_PATH / "owt.pkl", "wb") as f:
        pickle.dump({"vocab": vocab, "merges": merges}, f)


class BPETokenizer:
    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ):
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens or []
        self.should_handle_special_tokens = False
        if special_tokens:
            self.should_handle_special_tokens = True

        self.bytes_to_ids: dict[bytes, int] = {v: k for k, v in vocab.items()}
        self.special_tokens_bytes_to_ids: dict[bytes, int] = {}
        self.pretokenize_pat = (
            r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        )
        self.decoded_bytes = {}  # cache decoded bytes
        if self.should_handle_special_tokens:
            next_idx = max(vocab.keys()) + 1
            for t in self.special_tokens:
                tt = t.encode("utf-8")
                if tt not in self.bytes_to_ids:
                    self.bytes_to_ids[tt] = next_idx
                    self.vocab[next_idx] = tt
                self.special_tokens_bytes_to_ids[tt] = self.bytes_to_ids[tt]
            # ensures longer tokens come first to handle overlapping tokens
            # in case of ["<|endoftext|>", "<|endoftext|><|endoftext|>"]
            sorted_tokens = sorted(special_tokens, key=len, reverse=True)
            self.special_tokens_escaped = [re.escape(t) for t in sorted_tokens]
            self.special_token_split_pattern = f"({'|'.join(self.special_tokens_escaped)})"

    @classmethod
    def from_files(cls, file_path: str | os.PathLike, special_tokens: list[str] | None = None):
        with open(file_path, "rb") as f:
            d = pickle.load(f)
        return cls(vocab=d["vocab"], merges=d["merges"], special_tokens=special_tokens)

    def encode(self, text: str) -> list[int]:
        pretokenized_bytes = self.pretokenize(text)
        encoded_results = []
        for byte_word in tqdm(pretokenized_bytes, desc="Encoding", disable=not SHOW_PROGRESS):
            # byte_word = b'Hello'
            if byte_word in self.decoded_bytes:
                encoded_results.extend(self.decoded_bytes[byte_word])
            elif byte_word in self.special_tokens_bytes_to_ids:
                encoded_results.append(self.special_tokens_bytes_to_ids[byte_word])
            else:
                byte_seq = [bytes([i]) for i in list(byte_word)]
                for merge in self.merges:
                    byte_seq = self.apply_merge(byte_seq, merge)
                ids = [self.bytes_to_ids[b] for b in byte_seq]
                self.decoded_bytes[byte_word] = ids
                encoded_results.extend(ids)
        return encoded_results

    def encode_iterable(self, iter: Iterable[str]) -> Iterable[int]:
        for line in iter:
            encoded_line = self.encode(line)
            yield from encoded_line

    def decode(self, ids: list[int]) -> str:
        byte_list = [self.vocab[i] for i in ids]
        return b"".join(byte_list).decode("utf-8", errors="replace")

    def pretokenize(self, text: str) -> list[bytes]:
        pretokens = []
        if self.should_handle_special_tokens:
            parts = [p for p in re.split(self.special_token_split_pattern, text) if p]
        else:
            parts = [text]

        for part in parts:
            if part in self.special_tokens:
                pretokens.append(part.encode("utf-8"))
            else:
                for t in regex.finditer(self.pretokenize_pat, part):
                    pretokens.append(t.group().encode("utf-8"))
        return pretokens

    def apply_merge(self, byte_seq: list[bytes], merge: tuple[bytes, bytes]) -> list[bytes]:
        new_byte_seq = []
        i = 0
        while i < len(byte_seq):
            if i < len(byte_seq) - 1 and (byte_seq[i], byte_seq[i + 1]) == merge:
                new_byte_seq.append(byte_seq[i] + byte_seq[i + 1])
                i += 2
            else:
                new_byte_seq.append(byte_seq[i])
                i += 1
        return new_byte_seq


if __name__ == "__main__":
    input_file = OUTPUT_PATH / "tinystories.pkl"
    tokenizer = BPETokenizer.from_files(input_file, ["<|endoftext|>"])
    all_ids = []
    with open(FIXUTRES_PATH / "tinystories_sample.txt") as f:
        for _id in tokenizer.encode_iterable(f):
            all_ids.append(_id)
            print(_id)
