import random
import numpy as np
import torch
from tokenizer import BPETokenizer
import pathlib

DATA_PATH = (pathlib.Path(__file__).resolve().parent.parent) / "data"
OUTPUT_PATH = (pathlib.Path(__file__).resolve().parent.parent) / "bpe_output"


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def tokenize_tiny_stories():
    tokenizer_parmas = OUTPUT_PATH / "tinystories.pkl"
    tokenizer = BPETokenizer.from_files(tokenizer_parmas, ["<|endoftext|>"])
    print("tokenizing TinyStoriesV2-GPT4-valid")
    with open(DATA_PATH / "TinyStoriesV2-GPT4-valid.txt") as f:
        text_contents = f.read()
        ids = tokenizer.encode(text_contents)
        ids_arr = np.array(ids, dtype="int64")
        print(ids_arr[:1024])
        print(ids_arr.shape)
        print("max id", ids_arr.max())
        np.save(DATA_PATH / "TinyStoriesV2-GPT4-valid.npy", ids_arr)
    print("tokenizing TinyStoriesV2-GPT4-train")
    with open(DATA_PATH / "TinyStoriesV2-GPT4-train.txt") as f:
        text_contents = f.read()
        ids = tokenizer.encode(text_contents)
        ids_arr = np.array(ids, dtype="int64")
        print(ids_arr[:1024])
        print(ids_arr.shape)
        print("max id", ids_arr.max())
        np.save(DATA_PATH / "TinyStoriesV2-GPT4-train.npy", ids_arr)


if __name__ == "__main__":
    tokenize_tiny_stories()
