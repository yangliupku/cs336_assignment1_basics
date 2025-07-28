from collections.abc import Callable, Iterable
import torch
import math


class AdamW(torch.optim.Optimizer):
    def __init__(
        self,
        params,
        lr: float | torch.Tensor = 1e-3,
        betas: tuple[float | torch.Tensor, float | torch.Tensor] = (0.9, 0.99),
        eps: float = 1e-8,
        weight_decay: float | torch.Tensor = 0.01,
    ):
        defaults = {"lr": lr, "betas": betas, "eps": eps, "weight_decay": weight_decay}
        super().__init__(params, defaults)

    def step(self, clousure: Callable | None = None):
        loss = None
        if clousure is not None:
            loss = clousure()
        for group in self.param_groups:
            lr = group["lr"]
            betas = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                t = state.get("t", 1)
                m = state.get("m", torch.zeros(p.shape, device=p.device))
                v = state.get("v", torch.zeros(p.shape, device=p.device))
                g = p.grad.data
                m = betas[0] * m + (1 - betas[0]) * g
                v = betas[1] * v + (1 - betas[1]) * (g**2)
                a = lr * math.sqrt(1 - betas[1] ** t) / (1 - betas[0] ** t)
                p.data -= a * m / (torch.sqrt(v) + eps)
                p.data -= lr * weight_decay * p.data
                state["t"] = t + 1
                state["m"] = m
                state["v"] = v
        return loss


def lr_cosine_schedule(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
) -> float:
    assert it >= 0

    if it < warmup_iters:
        return it / warmup_iters * max_learning_rate
    elif it >= warmup_iters and it <= cosine_cycle_iters:
        return min_learning_rate + 0.5 * (
            1 + math.cos((it - warmup_iters) / (cosine_cycle_iters - warmup_iters) * math.pi)
        ) * (max_learning_rate - min_learning_rate)
    else:
        return min_learning_rate


def apply_gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float) -> None:
    eps = 1e-6
    l2 = 0
    for p in parameters:
        if p.grad is None:
            continue
        l2 += (p.grad.data**2).sum()
    l2 = math.sqrt(l2)
    if l2 >= max_l2_norm:
        a = max_l2_norm / (l2 + eps)
        for p in parameters:
            if p.grad is None:
                continue
            p.grad.data *= a


if __name__ == "__main__":
    weights = torch.nn.Parameter(torch.randn(10, 10))
    opt = AdamW([weights], lr=1.0)
    for i in range(10):
        opt.zero_grad()
        loss = (weights**2).mean()
        loss.backward()
        print(loss.cpu().item())
        opt.step()
