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
) -> list[str]:
    with open(input_path) as f:
        raw_text = f.read()
    special_tokens_escaped = [re.escape(t) for t in special_tokens]
    pattern = "|".join(special_tokens_escaped)
    parts = re.split(pattern, raw_text)
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    tokens = []
    for part in parts:
        tokens.extend(regex.findall(PAT, part))
    return tokens


if __name__ == "__main__":
    input_file = DATA_PATH / "example.txt"
    tokens = pretokenize(input_file)
    print(tokens)
