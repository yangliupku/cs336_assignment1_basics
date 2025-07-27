import torch
import math
from einops import einsum, rearrange, reduce
from jaxtyping import Float, Int


class Linear(torch.nn.Module):
    def __init__(self, in_features: int, out_features: int, device=None, dtype=None):
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


def cross_entropy(
    logits: Float[torch.Tensor, " batch_size vocab_size"], targets: Int[torch.Tensor, " batch_size"]
) -> Float[torch.Tensor, ""]:
    logits = logits.view(-1, logits.size(-1))
    targets = targets.view(-1)
    normalized_logits = logits - torch.max(logits, dim=-1, keepdim=True).values
    batch_indices = torch.arange(len(targets))
    ce = (
        torch.log(torch.sum(torch.exp(normalized_logits), dim=-1, keepdim=True))
        - normalized_logits[batch_indices, targets]
    )
    return torch.mean(ce)


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


class MultiHeadSelfAttention(torch.nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        max_seq_len: int = 1024,
        enable_rope: bool = False,
        theta: float = 1.0,
        device=None,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        # d_k = dv = d_model//num_heads
        self.q_proj_weight = torch.nn.Parameter(torch.empty(d_model, d_model, device=device))
        self.k_proj_weight = torch.nn.Parameter(torch.empty(d_model, d_model, device=device))
        self.v_proj_weight = torch.nn.Parameter(torch.empty(d_model, d_model, device=device))
        self.o_proj_weight = torch.nn.Parameter(torch.empty(d_model, d_model, device=device))
        sig = math.sqrt(1 / (d_model))
        for t in [self.q_proj_weight, self.k_proj_weight, self.v_proj_weight, self.o_proj_weight]:
            torch.nn.init.trunc_normal_(t, std=sig, a=-3, b=3)
        tri_mask = torch.tril(torch.ones(max_seq_len, max_seq_len)) == 1
        self.register_buffer("tri_mask", tri_mask)
        self.enable_rope = enable_rope
        self.rope = RotaryPositionalEmbedding(
            theta=theta, d_k=d_model // num_heads, max_seq_len=max_seq_len
        )

    def forward(
        self,
        x: Float[torch.Tensor, "... seq_length d_in"],
        token_positions: Int[torch.Tensor, "... seq_length"] | None = None,
    ) -> Float[torch.Tensor, "... seq_len d_out"]:
        seq_len = x.shape[-2]
        mask = self.tri_mask[:seq_len, :seq_len]
        if token_positions is None:
            token_positions = torch.arange(seq_len)
        q_proj_weight_batched = rearrange(
            self.q_proj_weight, "(head dh) dk -> head dh dk", head=self.num_heads
        )
        k_proj_weight_batched = rearrange(
            self.k_proj_weight, "(head dh) dk -> head dh dk", head=self.num_heads
        )
        v_proj_weight_batched = rearrange(
            self.v_proj_weight, "(head dh) dk -> head dh dk", head=self.num_heads
        )
        Q = einsum(x, q_proj_weight_batched, "... seq d, head dh d->... head seq dh")
        K = einsum(x, k_proj_weight_batched, "... seq d, head dh d->... head seq dh")
        V = einsum(x, v_proj_weight_batched, "... seq d, head dh d->... head seq dh")
        if self.enable_rope:
            Q = self.rope(Q, token_positions)
            K = self.rope(K, token_positions)

        att_res_batched = scaled_dot_product_attention(Q, K, V, mask)
        att_res = rearrange(att_res_batched, "... heads seq v-> ... seq (heads v)")
        return einsum(att_res, self.o_proj_weight, "... seq h, d h->... seq d")


class TransformerBlock(torch.nn.Module):
    def __init__(
        self, d_model: int, num_heads: int, d_ff: int, max_seq_len: int = 1024, theta: float = 1.0
    ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.rms_norm_1 = RMSNorm(d_model=d_model)
        self.rms_norm_2 = RMSNorm(d_model=d_model)
        self.attn = MultiHeadSelfAttention(
            d_model=d_model,
            num_heads=num_heads,
            max_seq_len=max_seq_len,
            enable_rope=True,
            theta=theta,
        )
        self.ffn = SwiGLUFF(d_model=d_model, d_ff=d_ff)

    def forward(self, x):
        x = x + self.attn(self.rms_norm_1(x))
        x = x + self.ffn(self.rms_norm_2(x))
        return x


class TransformerLM(torch.nn.Module):
    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        rope_theta: float,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.d_model = d_model
        self.num_layer = num_layers
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.rope_theta = rope_theta
        self.token_embeddings = Embedding(vocab_size, d_model)
        self.layers = torch.nn.ModuleList(
            [
                TransformerBlock(
                    d_model=d_model,
                    num_heads=num_heads,
                    d_ff=d_ff,
                    max_seq_len=context_length,
                    theta=rope_theta,
                )
                for i in range(num_layers)
            ]
        )
        self.ln_final = RMSNorm(d_model=d_model)
        self.lm_head = Linear(in_features=d_model, out_features=vocab_size)

    def forward(
        self, in_indices: Int[torch.Tensor, "batch seq"]
    ) -> Float[torch.Tensor, "batch seq vocab"]:
        """Out put logits before softmax"""
        x: Float[torch.Tenor, "batch seq d_model"] = self.token_embeddings(in_indices)
        for layer in self.layers:
            x = layer(x)
        x: Float[torch.Tensor, "batch seq d_model"] = self.ln_final(x)
        x: Float[torch.Tensor, "batch seq vocab"] = self.lm_head(x)
        return x


def generate(
    model: torch.nn.Module,
    prompt: list[int],
    context_length: int,
    max_new_tokens: int,
    temperature: float = 1.0,
    top_p: float = None,
):
    input_ids = prompt[-context_length:]
    in_indices = torch.tensor(input_ids).unsqueeze(0)
    out_ids = []
    for _ in range(max_new_tokens):
        logits = model(in_indices)
        logits = logits[:, -1, :] / temperature
        probs = stable_softmax(logits, dim=-1)

        if top_p is not None:
            sorted_probs, sorted_indices = torch.sort(probs, dim=-1, descending=True)
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
            # Find the smallest number of items that sum to at least p
            mask = cumulative_probs < top_p
            # offset mask by 1 to include the element that make prob sum > p
            mask = torch.cat([torch.ones_like(mask[:, :1]), mask[:, :-1]], dim=1)
            # Set probabilities of excluded items to 0
            sorted_probs[~mask] = 0
            # Re-normalize probabilities
            probs = torch.zeros_like(probs).scatter_(-1, sorted_indices, sorted_probs)

        id_next = torch.multinomial(probs, num_samples=1)
        out_ids.append(id_next[0, 0].item())
        if id_next[0, 0].item() == 0:
            break
        in_indices = torch.cat((in_indices, id_next), dim=1)[:, -context_length:]
    return out_ids


if __name__ == "__main__":
    seed = 0
    torch.manual_seed(seed)
    vocab_size = 20
    batch_size = 5
    embedding_size = 8
    max_seq_length = 10
    num_heads = 2
    num_layers = 2
    seq_len = 3
    # input = torch.randint(0, vocab_size, (batch_size, seq_len))
    model = TransformerLM(
        vocab_size=vocab_size,
        context_length=max_seq_length,
        d_model=embedding_size,
        num_layers=num_layers,
        num_heads=num_heads,
        d_ff=4 * embedding_size,
        rope_theta=1.0,
    )
    outputs = generate(
        model=model,
        prompt=[3, 8, 2, 5, 1],
        context_length=max_seq_length,
        max_new_tokens=100,
        temperature=1.0,
        top_p=0.8,
    )
    print(outputs)
