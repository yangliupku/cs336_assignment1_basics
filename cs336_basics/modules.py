import torch
import math
from einops import einsum, rearrange, reduce
from jaxtyping import Float


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


class SiLU(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(x)


class RotaryPositionalEmbedding(torch.nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()
        self.theta = theta
        self.d_k = d_k
        assert d_k % 2 == 0
        self.max_seq_len = max_seq_len
        rot_matrix: Float[torch.Tensor, "max_seq_len d_k/2 2 2"] = self.get_rotation_maxtrix().to(
            device
        )
        self.register_buffer("rot_matrix", rot_matrix, persistent=False)

    def get_rotation_maxtrix(self) -> Float[torch.Tensor, "max_seq_len d_k/2 2 2"]:
        dk_2 = self.d_k // 2
        d_k = self.d_k
        i = torch.arange(0, self.max_seq_len)
        k = torch.arange(0, dk_2)
        theta_ik = einsum(i, 1 / self.theta ** (2 * k / d_k), "s, dk2 -> s dk2")
        r = [
            torch.cos(theta_ik),
            -1 * torch.sin(theta_ik),
            torch.sin(theta_ik),
            torch.cos(theta_ik),
        ]
        return rearrange(r, "(d1 d2) s dk2 -> s dk2 d1 d2", d1=2)

    def forward(
        self,
        x: Float[torch.Tensor, "... seq_len d_k"],
        token_positions: Float[torch.Tensor, "... seq_len"],
    ) -> Float[torch.Tensor, "... seq_len d_k"]:
        rot: Float[torch.Tensor, "... seq_len d_k/2 2 2"] = self.rot_matrix[token_positions]
        x1 = rearrange(x, "... (dk2 d2) -> ... dk2 d2", d2=2)
        x2 = einsum(x1, rot, "... dk2 d2, ... dk2 d1 d2 -> ... dk2 d1")
        return rearrange(x2, "... dk2 d1-> ... (dk2 d1)")


class SwiGLUFF(torch.nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.w1_weight = torch.nn.Parameter(torch.empty(d_ff, d_model))
        self.w2_weight = torch.nn.Parameter(torch.empty(d_model, d_ff))
        self.w3_weight = torch.nn.Parameter(torch.empty(d_ff, d_model))
        self.silu = SiLU()
        sig = math.sqrt(2 / (d_model + d_model))
        for t in [self.w1_weight, self.w2_weight, self.w3_weight]:
            torch.nn.init.trunc_normal_(t, std=sig, a=-3, b=3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a = einsum(x, self.w1_weight, "... d_model, d_ff d_model -> ... d_ff")
        b = einsum(x, self.w3_weight, "... d_model, d_ff d_model -> ... d_ff")
        return einsum(self.silu(a) * b, self.w2_weight, "... d_ff, d_model d_ff -> ... d_model")


def stable_softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    c = x - torch.max(x, dim=dim, keepdim=True).values  # subtract max to normalize
    c = torch.exp(c)
    return c / torch.sum(c, dim=dim, keepdim=True)


def scaled_dot_product_attention(
    Q: Float[torch.Tensor, "batch ... seq_len d_k"],
    K: Float[torch.Tensor, "batch ... seq_len d_k"],
    V: Float[torch.Tensor, "batch ... seq_len d_v"],
    mask: Float[torch.Tensor, "seq_len seq_len"],
) -> Float[torch.Tensor, "batch ... seq_len dv"]:
    att_map = einsum(Q, K, "... q dk, ... k dk -> ... q k") / math.sqrt(Q.shape[-1])
    masked_att_map = att_map.masked_fill(~mask, float("-inf"))
    sm_att_map: Float[torch.Tensor, "... q k"] = stable_softmax(masked_att_map, dim=-1)
    return einsum(sm_att_map, V, "... q s, ... s v -> ... q v")


if __name__ == "__main__":
    seed = 0
    torch.manual_seed(seed)
    batch_size = 2
    embedding_size = 4
    max_seq_length = 10
    seq_len = 3
    input = torch.rand(seq_len, embedding_size)
    # token_positions = torch.randint(0, max_seq_length - 1, (batch_size, seq_len))
    # layer = RotaryPositionalEmbedding(theta=0.5, d_k=embedding_size, max_seq_len=max_seq_length)
    # print(layer(input, token_positions))
    print(stable_softmax(input, -1))
