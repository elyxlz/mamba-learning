import dataclasses
import typing
import torch
import einops as eo


@dataclasses.dataclass
class Mamba2Config:
    input_dim: int = 16
    hidden_dim: int = 64  # model dimension (D)
    num_layers: int = 2  # number of Mamba-2 layers in the language model
    state_dim: int = 16  # 128  # state dimension (N)
    conv_dim: int = 4  # convolution kernel size
    expand: int = 2  # expansion factor (E)
    head_dim: int = 8  # 64  # head dimension (P)
    chunk_size: int = 8  # 64  # matrix partition size (Q)

    def __post_init__(self):
        self.inner_dim = self.expand * self.hidden_dim
        assert self.inner_dim % self.head_dim == 0
        self.nheads = self.inner_dim // self.head_dim


def segsum(x: torch.Tensor) -> torch.Tensor:
    T = x.size(-1)
    x = eo.repeat(x, "... d -> ... d e", e=T)
    mask = torch.tril(torch.ones(T, T, dtype=torch.bool, device=x.device), diagonal=-1)
    x = x.masked_fill(~mask, 0)
    x_segsum = torch.cumsum(x, dim=-2)
    mask = torch.tril(torch.ones(T, T, dtype=torch.bool, device=x.device), diagonal=0)
    x_segsum = x_segsum.masked_fill(~mask, -torch.inf)
    return x_segsum


def ssd(x, A, B, C, chunk_size, initial_states=None):
    assert x.shape[1] % chunk_size == 0

    x, A, B, C = [
        eo.rearrange(m, "b (c l) ... -> b c l ...", l=chunk_size) for m in (x, A, B, C)
    ]

    A = eo.rearrange(A, "b c l h -> b h c l")
    A_cumsum = torch.cumsum(A, dim=-1)

    L = torch.exp(segsum(A))
    Y_diag = torch.einsum("bclhn, bcshn, bhcls, bcshp -> bclhp", C, B, L, x)

    decay_states = torch.exp(A_cumsum[:, :, :, -1:] - A_cumsum)
    states = torch.einsum("bclhn, bhcl, bclhp -> bchpn", B, decay_states, x)

    if initial_states is None:
        initial_states = torch.zeros_like(states[:, :1])
    states = torch.cat([initial_states, states], dim=1)
    decay_chunk = torch.exp(
        segsum(torch.nn.functional.pad(A_cumsum[:, :, :, -1], (1, 0)))
    )
    new_states = torch.einsum("bhzc, bchpn -> bzhpn", decay_chunk, states)
    states, final_state = new_states[:, :-1], new_states[:, -1]

    # 4. Compute state -> output conversion per chunk
    # (left term of low-rank factorization of off-diagonal blocks; C terms)
    state_decay_out = torch.exp(A_cumsum)
    Y_off = torch.einsum("bclhn, bchpn, bhcl -> bclhp", C, states, state_decay_out)

    # Add output of intra-chunk and inter-chunk terms (diagonal and off-diagonal blocks)
    Y = eo.rearrange(Y_diag + Y_off, "b c l h p -> b (c l) h p")

    return Y, final_state


class RMSNorm(torch.nn.Module):
    def __init__(self, d: int, eps: float = 1e-5):
        """Gated Root Mean Square Layer Normalization

        Paper: https://arxiv.org/abs/1910.07467
        """
        super().__init__()
        self.eps = eps
        self.weight = torch.nn.Parameter(torch.ones(d))

    def forward(self, x, z=None):
        if z is not None:
            x = x * torch.nn.functional.silu(z)
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


class InferenceCache(typing.NamedTuple):
    conv_state: torch.Tensor  # (batch, inner_dim + 2 * state_dim, conv_dim)
    ssm_state: torch.Tensor  # (batch, nheads, head_dim, state_dim)

    @staticmethod
    def alloc(
        batch_size: int, config: Mamba2Config, device: torch.device | None = None
    ):
        return InferenceCache(
            torch.zeros(
                batch_size,
                config.inner_dim + 2 * config.state_dim,
                config.conv_dim,
                device=device,
            ),
            torch.zeros(
                batch_size,
                config.nheads,
                config.head_dim,
                config.state_dim,
                device=device,
            ),
        )


class Mamba2(torch.nn.Module):
    def __init__(self, config: Mamba2Config):
        super().__init__()
        self.config = config

        # Order: (z, x, B, C, dt)
        d_in_proj = 2 * config.inner_dim + 2 * config.state_dim + config.nheads
        self.in_proj = torch.nn.Linear(config.hidden_dim, d_in_proj, bias=False)

        conv_dim = config.inner_dim + 2 * config.state_dim
        self.conv1d = torch.nn.Conv1d(
            conv_dim,
            out_channels=conv_dim,
            kernel_size=config.conv_dim,
            groups=conv_dim,
            padding=config.conv_dim - 1,
        )

        self.dt_bias = torch.nn.Parameter(torch.empty(config.nheads))
        self.A_log = torch.nn.Parameter(torch.empty(config.nheads))
        self.D = torch.nn.Parameter(torch.empty(config.nheads))
        self.norm = RMSNorm(config.inner_dim)
        self.out_proj = torch.nn.Linear(config.inner_dim, config.hidden_dim, bias=False)

    def forward(self, u: torch.Tensor, h: InferenceCache | None = None):
        if h:
            return self.step(u, h)

        A = -torch.exp(self.A_log)  # (nheads,)
        zxbcdt = self.in_proj(u)  # (batch, seqlen, d_in_proj)
        z, xBC, dt = torch.split(
            zxbcdt,
            [
                self.config.inner_dim,
                self.config.inner_dim + 2 * self.config.state_dim,
                self.config.nheads,
            ],
            dim=-1,
        )
        dt = torch.nn.functional.softplus(dt + self.dt_bias)  # (batch, seqlen, nheads)

        # Pad or truncate xBC seqlen to conv_dim
        conv_state = torch.nn.functional.pad(
            eo.rearrange(xBC, "b l d -> b d l"), (self.config.conv_dim - u.shape[1], 0)
        )

        xBC = torch.nn.functional.silu(
            self.conv1d(xBC.transpose(1, 2)).transpose(1, 2)[:, : u.shape[1], :]
        )  # (batch, seqlen, inner_dim + 2 * state_dim))
        x, B, C = torch.split(
            xBC,
            [self.config.inner_dim, self.config.state_dim, self.config.state_dim],
            dim=-1,
        )
        x = eo.rearrange(x, "b l (h p) -> b l h p", p=self.config.head_dim)
        y, ssm_state = ssd(
            x * dt.unsqueeze(-1),
            A * dt,
            eo.rearrange(B, "b l n -> b l 1 n"),
            eo.rearrange(C, "b l n -> b l 1 n"),
            self.config.chunk_size,
        )
        y = y + x * self.D.unsqueeze(-1)
        y = eo.rearrange(y, "b l h p -> b l (h p)")
        y = self.norm(y, z)
        y = self.out_proj(y)

        h = InferenceCache(conv_state, ssm_state)
        return y, h

    def step(
        self, u: torch.Tensor, h: InferenceCache
    ) -> tuple[torch.Tensor, InferenceCache]:
        assert u.shape[1] == 1, "Only one token can be decoded per inference step"

        zxbcdt = self.in_proj(u.squeeze(1))  # (batch, d_in_proj)
        z, xBC, dt = torch.split(
            zxbcdt,
            [
                self.config.inner_dim,
                self.config.inner_dim + 2 * self.config.state_dim,
                self.config.nheads,
            ],
            dim=-1,
        )

        # Advance convolution input
        h.conv_state.copy_(torch.roll(h.conv_state, shifts=-1, dims=-1))
        h.conv_state[:, :, -1] = xBC
        # Convolution step
        xBC = torch.sum(
            h.conv_state * eo.rearrange(self.conv1d.weight, "d 1 w -> d w"), dim=-1
        )
        xBC += self.conv1d.bias
        xBC = torch.nn.functional.silu(xBC)

        x, B, C = torch.split(
            xBC,
            [self.config.inner_dim, self.config.state_dim, self.config.state_dim],
            dim=-1,
        )
        A = -torch.exp(self.A_log)  # (nheads,)

        # SSM step
        dt = torch.nn.functional.softplus(dt + self.dt_bias)  # (batch, nheads)
        dA = torch.exp(dt * A)  # (batch, nheads)
        x = eo.rearrange(x, "b (h p) -> b h p", p=self.config.head_dim)
        dBx = torch.einsum("bh, bn, bhp -> bhpn", dt, B, x)
        h.ssm_state.copy_(h.ssm_state * eo.rearrange(dA, "b h -> b h 1 1") + dBx)
        y = torch.einsum("bhpn, bn -> bhp", h.ssm_state, C)
        y = y + eo.rearrange(self.D, "h -> h 1") * x
        y = eo.rearrange(y, "b h p -> b (h p)")
        y = self.norm(y, z)
        y = self.out_proj(y)

        return y.unsqueeze(1), h


class Mamba2LMHeadModel(torch.nn.Module):
    def __init__(self, config: Mamba2Config) -> None:
        super().__init__()
        self.config = config

        self.backbone = torch.nn.ModuleDict(
            dict(
                embedding=torch.nn.Embedding(config.vocab_size, config.hidden_dim),
                layers=torch.nn.ModuleList(
                    [
                        torch.nn.ModuleDict(
                            dict(
                                mixer=Mamba2(
                                    config,
                                ),
                                norm=RMSNorm(
                                    config.hidden_dim,
                                ),
                            )
                        )
                        for _ in range(config.num_layers)
                    ]
                ),
                norm_f=RMSNorm(config.hidden_dim),
            )
        )
        self.head = torch.nn.Linear(config.hidden_dim, config.input_dim * 2, bias=False)

    def forward(
        self,
        input_ids: torch.Tensor,
        h: list[InferenceCache] | list[None] | None = None,
    ) -> tuple[torch.Tensor, list[InferenceCache]]:
        seqlen = input_ids.shape[1]

        if h is None:
            h = [None for _ in range(self.config.num_layers)]

        x = self.backbone.embedding(input_ids)
        for i, layer in enumerate(self.backbone.layers):
            y, h[i] = layer.mixer(layer.norm(x), h[i])
            x = y + x

        x = self.backbone.norm_f(x)
        logits = self.lm_head(x)
        return logits[:, :seqlen], typing.cast(list[InferenceCache], h)

    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_length: int = 20,
    ) -> typing.Iterable[tuple[int, list[InferenceCache]]]:
        prefix, tokens = input_ids[:-1], input_ids[-1:].unsqueeze(0)

        # Process prompt
        # The input sequence to forward (non-inference path) must have length multiple that of chunk_size.
        # We split out excess tokens so that n_chunked tokens can be processed by one forward call and
        # process the rest in multiple inference steps.
        n_chunked = (prefix.shape[0] // self.config.chunk_size) * self.config.chunk_size
        if n_chunked > 0:
            _, h = self(prefix[:n_chunked].unsqueeze(0), None)
        else:
            h = [
                InferenceCache.alloc(1, self.config, device=self.device)
                for _ in range(self.config.num_layers)
            ]
        for i in range(n_chunked, prefix.shape[0]):
            _, h = self(prefix[i : i + 1].unsqueeze(0), h)

        # Generate
        for _ in range(max_new_length):
            with torch.no_grad():
                out, h = self(tokens, h)
            logits = out[0, -1]
            probs = torch.nn.functional.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            tokens = next_token.unsqueeze(0)
            yield typing.cast(int, next_token.item()), h
