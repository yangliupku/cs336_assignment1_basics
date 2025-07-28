import hydra
from omegaconf import DictConfig
from cs336_basics.modules import TransformerLM, cross_entropy
from cs336_basics.utils import set_seed
from cs336_basics.data import get_batch
import numpy as np
from cs336_basics.optimizers import AdamW


@hydra.main(config_path="configs", config_name="config")
def train(cfg: DictConfig):
    device = "mps"
    set_seed(cfg.seed)
    model = TransformerLM(
        **cfg.model,
    )
    model.to(device)
    test_data = np.random.randint(0, cfg.model.vocab_size, 1000)
    test_batch_inputs, test_batch_labels = get_batch(
        test_data, cfg.training.batch_size, cfg.model.context_length, device
    )
    opt = AdamW(
        params=model.parameters(),
        lr=cfg.optimizer.lr,
        betas=tuple(cfg.optimizer.beta),
        weight_decay=cfg.optimizer.weight_decay,
    )
    for i in range(100):
        opt.zero_grad()
        logits = model(test_batch_inputs)
        loss = cross_entropy(logits, test_batch_labels)
        loss.backward()
        print(loss.item())
        opt.step()


if __name__ == "__main__":
    train()
