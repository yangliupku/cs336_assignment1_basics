import hydra
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm
from cs336_basics.modules import TransformerLM, cross_entropy
from cs336_basics.utils import set_seed, generate_experiment_name, create_experiment_structure
from cs336_basics.data import get_batch, estimate_loss, save_checkpoint
import numpy as np
from cs336_basics.optimizers import AdamW
import pathlib


DATA_PATH = (pathlib.Path(__file__).resolve().parent) / "data"


@hydra.main(config_path="configs", config_name="config")
def train(cfg: DictConfig):
    device = "mps"
    set_seed(cfg.seed)
    if cfg.experiment.name is None:
        exp_name = generate_experiment_name(cfg)
    else:
        exp_name = cfg.experiment.name
    print(exp_name)
    exp_path = create_experiment_structure(exp_name)
    OmegaConf.save(cfg, f"{exp_path}/configs/config.yaml")

    model = TransformerLM(
        vocab_size=cfg.model.vocab_size,
        context_length=cfg.model.context_length,
        d_model=cfg.model.d_model,
        num_layers=cfg.model.num_layers,
        num_heads=cfg.model.num_heads,
        d_ff=cfg.model.d_ff,
        rope_theta=cfg.model.rope_theta,
    )
    model.to(device)
    if cfg.dataset == "Tinystory":
        train_data = np.load(DATA_PATH / "TinyStoriesV2-GPT4-train.npy", mmap_mode="r")
        eval_data = np.load(DATA_PATH / "TinyStoriesV2-GPT4-valid.npy", mmap_mode="r")
    else:
        raise NotImplementedError

    opt = AdamW(
        params=model.parameters(),
        lr=cfg.optimizer.lr,
        betas=tuple(cfg.optimizer.beta),
        weight_decay=cfg.optimizer.weight_decay,
    )
    for iter in tqdm(range(cfg.training.max_iters), desc="training"):
        train_batch_inputs, train_batch_labels = get_batch(
            train_data, cfg.training.batch_size, cfg.model.context_length, device
        )
        opt.zero_grad()
        logits = model(train_batch_inputs)
        loss = cross_entropy(logits, train_batch_labels)
        loss.backward()
        opt.step()
        if iter % cfg.training.log_every == 0:
            train_loss = estimate_loss(
                train_data,
                model,
                cfg.training.batch_size,
                cfg.model.context_length,
                cfg.training.eval_iters,
                device,
            )
            eval_loss = estimate_loss(
                eval_data,
                model,
                cfg.training.batch_size,
                cfg.model.context_length,
                cfg.training.eval_iters,
                device,
            )
            tqdm.write(f"iter: {iter}, train_loss={train_loss}, eval_loss={eval_loss}")
        if iter % cfg.training.save_every == 0:
            checkpoint_path = exp_path / f"checkpoints/iter_{iter}.pt"
            save_checkpoint(model, opt, iter, checkpoint_path)
            tqdm.write(f"save checkpoint iter_{iter}.pt")


if __name__ == "__main__":
    train()
