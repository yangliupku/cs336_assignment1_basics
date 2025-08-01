from hydra import initialize_config_dir, compose
from cs336_basics.modules import TransformerLM, generate
from cs336_basics.utils import set_seed
from cs336_basics.data import load_checkpoint
from cs336_basics.optimizers import AdamW
import pathlib
from cs336_basics.tokenizer import BPETokenizer


DATA_PATH = (pathlib.Path(__file__).resolve().parent.parent) / "data"
EXP_PATH = (pathlib.Path(__file__).resolve().parent.parent) / "experiments"
BPE_PATH = (pathlib.Path(__file__).resolve().parent.parent) / "bpe_output"


def eval(exp_name: str):
    checkpoint_folder = EXP_PATH / exp_name / "checkpoints"
    config_folder = EXP_PATH / exp_name / "configs"
    print(config_folder)
    with initialize_config_dir(config_dir=str(config_folder.absolute()), version_base=None):
        cfg = compose(config_name="config")
    device = "mps"
    set_seed(cfg.seed)

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

    opt = AdamW(
        params=model.parameters(),
        lr=cfg.optimizer.lr,
        betas=tuple(cfg.optimizer.beta),
        weight_decay=cfg.optimizer.weight_decay,
    )
    load_checkpoint(
        src=checkpoint_folder / "iter_4500.pt",
        model=model,
        optimizer=opt,
    )
    tokenizer_parmas = BPE_PATH / "tinystories.pkl"
    tokenizer = BPETokenizer.from_files(tokenizer_parmas, ["<|endoftext|>"])
    prompts = [
        "In the bustling city of New York, a taxi driver noticed",
        "Once upon a time, in a land far, far away, there lived a",
        "I have a dream",
    ]
    for prompt in prompts:
        print("prompt:", prompt)
        inputs = tokenizer.encode(prompt)
        outputs = generate(
            model=model,
            prompt=inputs,
            context_length=cfg.model.context_length,
            max_new_tokens=cfg.generate.max_new_tokens,
            temperature=cfg.generate.temperature,
            top_p=cfg.generate.top_p,
            device=device,
        )
        print("response:", tokenizer.decode(outputs))


if __name__ == "__main__":
    exp_name = "20250730_232338_TransformerLM-Tinystory_lr1e-03_beta0.9_0.99_bs32_wd1e-02_cs336_assignment1"
    eval(exp_name)
