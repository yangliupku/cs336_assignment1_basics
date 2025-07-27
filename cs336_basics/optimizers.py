from collections.abc import Callable
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
                m = state.get("m", torch.zeros(p.shape))
                v = state.get("v", torch.zeros(p.shape))
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


if __name__ == "__main__":
    weights = torch.nn.Parameter(torch.randn(10, 10))
    opt = AdamW([weights], lr=1.0)
    for i in range(10):
        opt.zero_grad()
        loss = (weights**2).mean()
        loss.backward()
        print(loss.cpu().item())
        opt.step()
