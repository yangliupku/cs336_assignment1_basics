from datetime import datetime
import random
import numpy as np
import torch
from cs336_basics.tokenizer import BPETokenizer
import pathlib
import os
from cs336_basics.logger import configure_global_logging

DATA_PATH = (pathlib.Path(__file__).resolve().parent.parent) / "data"
OUTPUT_PATH = (pathlib.Path(__file__).resolve().parent.parent) / "bpe_output"
EXP_PATH = (pathlib.Path(__file__).resolve().parent.parent) / "experiments"


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


def generate_experiment_name(cfg):
    """Generate a systematic experiment name"""

    # Timestamp for uniqueness and sorting
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Model info
    model_name = f"{cfg.model.name}-{cfg.dataset}"

    # Key hyperparameters (the ones that matter most)
    key_params = []
    beta = tuple(cfg.optimizer.beta)
    key_params.append(f"lr{cfg.optimizer.lr:.0e}")  # lr5e4
    key_params.append(f"beta{beta[0]}_{beta[1]}")  # beta(0.9,0.99)
    key_params.append(f"bs{cfg.training.batch_size}")  # bs32
    key_params.append(f"wd{cfg.optimizer.weight_decay:.0e}")

    # Join key params
    params_str = "_".join(key_params)

    # Description/tag
    description = cfg.experiment.description or "exp"

    # Combine all parts
    exp_name = f"{timestamp}_{model_name}_{params_str}_{description}"

    return exp_name


def create_experiment_structure(exp_name):
    """Create folder structure for experiment"""
    base_path = EXP_PATH / exp_name

    folders = [
        f"{base_path}/checkpoints",
        f"{base_path}/logs",
        f"{base_path}/outputs",
        f"{base_path}/configs",
    ]

    for folder in folders:
        os.makedirs(folder, exist_ok=True)

    return base_path


if __name__ == "__main__":
    tokenize_tiny_stories()
