import math
import torch
import dataclasses
import einops as eo


@dataclasses.dataclass
class MambaConfig:
    input_dim: int = 32
    hidden_dim: int = 64
    num_layers: int = 2
    state_dim: int = 16
    expand: int = 2
    dt_rank: int = -1
    d_conv: int = 4
    pad_vocab_size_multiple: int = 8
    conv_bias: bool = True
    bias: bool = False

    def __post_init__(self):
        self.d_inner = int(self.expand * self.hidden_dim)

        if self.dt_rank == -1:
            self.dt_rank = math.ceil(self.hidden_dim / 16)


class RMSNorm(torch.nn.Module):
    def __init__(self, hidden_dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = torch.nn.Parameter(torch.ones(hidden_dim))

    def forward(self, x):
        output = (
            x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight
        )

        return output


class ResidualBlock(torch.nn.Module):
    def __init__(self, config: MambaConfig):
        """Simple block wrapping Mamba block with normalization and residual connection."""
        super().__init__()
        self.config = config
        self.mixer = MambaBlock(config)
        self.norm = RMSNorm(config.hidden_dim)

    def forward(self, x):
        output = self.mixer(self.norm(x)) + x

        return output


class MambaBlock(torch.nn.Module):
    def __init__(self, config: MambaConfig):
        """A single Mamba block, as described in Figure 3 in Section 3.4 in the Mamba paper [1]."""
        super().__init__()
        self.config = config

        self.in_proj = torch.nn.Linear(
            config.hidden_dim, config.d_inner * 2, bias=config.bias
        )

        self.conv1d = torch.nn.Conv1d(
            in_channels=config.d_inner,
            out_channels=config.d_inner,
            bias=config.conv_bias,
            kernel_size=config.d_conv,
            groups=config.d_inner,
            padding=config.d_conv - 1,
        )

        # x_proj takes in `x` and outputs the input-specific Δ, B, C
        self.x_proj = torch.nn.Linear(
            config.d_inner, config.dt_rank + config.state_dim * 2, bias=False
        )

        # dt_proj projects Δ from dt_rank to d_in
        self.dt_proj = torch.nn.Linear(config.dt_rank, config.d_inner, bias=True)

        A = eo.repeat(
            torch.arange(1, config.state_dim + 1), "n -> d n", d=config.d_inner
        )
        self.A_log = torch.nn.Parameter(torch.log(A))
        self.D = torch.nn.Parameter(torch.ones(config.d_inner))
        self.out_proj = torch.nn.Linear(
            config.d_inner, config.hidden_dim, bias=config.bias
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s = x.size(1)

        x_and_res = self.in_proj(x)  # shape (b, s, 2 * d_in)
        (x, res) = x_and_res.split(
            split_size=[self.config.d_inner, self.config.d_inner], dim=-1
        )

        x = eo.rearrange(x, "b s d_in -> b d_in s")
        x = self.conv1d(x)[:, :, :s]
        x = eo.rearrange(x, "b d_in s -> b s d_in")

        x = torch.nn.functional.silu(x)

        y = self.ssm(x)

        y = y * torch.nn.functional.silu(res)

        output = self.out_proj(y)

        return output

    def ssm(self, x):
        (d_in, n) = self.A_log.shape

        A = -torch.exp(self.A_log.float())  # shape (d_in, n)
        D = self.D.float()

        x_dbl = self.x_proj(x)  # (b, s, dt_rank + 2*n)

        (delta, B, C) = x_dbl.split(
            split_size=[self.config.dt_rank, n, n], dim=-1
        )  # delta: (b, s, dt_rank). B, C: (b, s, n)
        delta = torch.nn.functional.softplus(self.dt_proj(delta))  # (b, s, d_in)

        y = self.selective_scan(
            x, delta, A, B, C, D
        )  # This is similar to run_SSM(A, B, C, u) in The Annotated S4 [2]

        return y

    def selective_scan(self, u, delta, A, B, C, D):
        (b, s, d_in) = u.shape
        n = A.shape[1]

        deltaA = torch.exp(eo.einsum(delta, A, "b s d_in, d_in n -> b s d_in n"))
        deltaB_u = eo.einsum(delta, B, u, "b s d_in, b s n, b s d_in -> b s d_in n")

        x = torch.zeros((b, d_in, n), device=deltaA.device)
        ys = []
        for i in range(s):
            x = deltaA[:, i] * x + deltaB_u[:, i]
            y = eo.einsum(x, C[:, i, :], "b d_in n, b n -> b d_in")
            ys.append(y)
        y = torch.stack(ys, dim=1)  # shape (b, s, d_in)

        y = y + u * D

        return y


class Mamba(torch.nn.Module):
    def __init__(self, config: MambaConfig):
        """Full Mamba model."""
        super().__init__()
        self.config = config

        self.in_proj = torch.nn.Linear(config.input_dim, out_features=config.hidden_dim)

        self.layers = torch.nn.ModuleList(
            [ResidualBlock(config) for _ in range(config.num_layers)]
        )
        self.norm_f = RMSNorm(config.hidden_dim)

        self.out_proj = torch.nn.Linear(
            config.hidden_dim, out_features=config.input_dim * 2
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.in_proj(x)

        for layer in self.layers:
            x = layer(x)

        x = self.norm_f(x)
        mean, var = self.out_proj(x).chunk(2, dim=-1)
        return torch.randn_like(mean) * var + mean


if __name__ == "__main__":
    model = Mamba(MambaConfig())

    seq = torch.randn((1, 64, 32))
    out = model(seq)
    print(out.shape)
