import os
import typing
import numpy as np
import torch
import numpy.typing as npt


def get_batch(
    dataset: npt.NDArray, batch_size: int, context_length: int, device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    assert len(dataset.shape) == 1
    text_size = len(dataset)
    starts = np.random.randint(0, text_size - context_length, size=batch_size)
    inputs = np.stack([dataset[start : start + context_length] for start in starts])
    labels = np.stack([dataset[start + 1 : start + context_length + 1] for start in starts])
    return (torch.tensor(inputs, device=device), torch.tensor(labels, device=device))


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | typing.BinaryIO | typing.IO[bytes],
) -> None:
    states = {
        "model_states": model.state_dict(),
        "optimizer_states": optimizer.state_dict(),
        "iteration": iteration,
    }
    torch.save(states, out)


def load_checkpoint(
    src: str | os.PathLike | typing.BinaryIO | typing.IO[bytes],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> int:
    states = torch.load(src)
    model.load_state_dict(states["model_states"])
    optimizer.load_state_dict(states["optimizer_states"])
    return states["iteration"]


if __name__ == "__main__":
    raw_ids = np.arange(1024)
    context_length = 64
    batch_size = 16
    device = "mps"
    (inputs, labels) = get_batch(raw_ids, batch_size, context_length, device)
    print(inputs)
