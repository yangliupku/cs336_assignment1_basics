import torch
import math
from einops import einsum, rearrange, reduce


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

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return einsum(input, self.weight, "... d_in, d_out d_in -> ... d_out")


class Embedding(torch.nn.Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.weight = torch.nn.Parameter(
            torch.empty((num_embeddings, embedding_dim), device=device, dtype=dtype)
        )
        # init parameter
        torch.nn.init.trunc_normal_(self.weight, a=-3, b=3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.weight[x]


class RMSNorm(torch.nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.weight = torch.nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x = x.to(torch.float32)
        rms = rearrange(reduce(x**2, "b s d -> b s", "mean"), "b s->b s 1")
        rms = torch.sqrt(rms + self.eps)
        result = x / rms * self.weight
        return result.to(in_dtype)


if __name__ == "__main__":
    # x = torch.rand(2, 5)
    # print(x)
    # layer = Linear(5, 4)
    # print(layer.weight)
    # print(layer(x))
    vocab_size = 10
    batch_size = 2
    embedding_size = 4
    seq_len = 3
    # embedding = Embedding(vocab_size, embedding_size)
    # input = torch.randint(0, vocab_size, (batch_size, seq_len))
    # x = embedding(input)
    # print(x)
    # assert x.shape == (batch_size, seq_len, embedding_size)
    input = torch.rand(batch_size, seq_len, embedding_size)
    layer = RMSNorm(embedding_size)
    print(layer(input))
