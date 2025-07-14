import torch
import math
from einops import einsum


class Linear(torch.nn.Module):
    def __init__(self, in_features, out_features, device=None, dtype=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = torch.nn.Parameter(
            torch.empty((out_features, in_features), device=device, dtype=dtype)
        )
        # init parameter
        sig = math.sqrt(2 / (in_features + out_features))
        torch.nn.init.trunc_normal_(self.weight, std=sig, a=-3, b=3)

    def forward(self, input):
        return einsum(input, self.weight, "... d_in, d_out d_in -> ... d_out")


if __name__ == "__main__":
    x = torch.rand(2, 5)
    print(x)
    layer = Linear(5, 4)
    print(layer.weight)
    print(layer(x))
