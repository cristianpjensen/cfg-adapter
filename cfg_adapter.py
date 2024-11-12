import math
import torch
import torch.nn as nn
from typing import Callable, Tuple, Optional


class SinusoidalEncoding(nn.Module):
    def __init__(self, hidden_dim: int, max_period: float=10000.0):
        super().__init__()
        assert hidden_dim % 2 == 0, "hidden_dim must be an even number"

        self.div_term = nn.Parameter(
            torch.exp(-math.log(max_period) * torch.arange(0, hidden_dim, 2, dtype=torch.float) / hidden_dim),
            requires_grad=False
        )
        self.hidden_dim = hidden_dim

    def forward(self, pos: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pos: ({B})
        
        Returns: ({B}, hidden_dim)
        """

        pos_div = torch.einsum("...b, d -> ...bd", pos.float(), self.div_term)
        return torch.cat([torch.sin(pos_div), torch.cos(pos_div)], dim=-1)


class CFGAdapterBlock(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: Optional[int]=None, mlp_ratio: int=4):
        super().__init__()

        self.output_dim = output_dim if output_dim is not None else input_dim

        self.hidden_dim = hidden_dim
        self.mlp_ratio = mlp_ratio
        self.cfg_encoder, cfg_dim = self.init_cfg_encoder()

        assert type(cfg_dim) == int, "cfg_dim must be an integer"
        assert cfg_dim > 0, "cfg_dim must be a positive integer"

        self.query_proj = nn.Linear(input_dim, hidden_dim, bias=False)
        self.key_proj = nn.Linear(cfg_dim, hidden_dim, bias=False)
        self.value_proj = nn.Linear(cfg_dim, self.output_dim, bias=False)
        self.scale = 1.0 / math.sqrt(hidden_dim)

        # Zero-initialize value projection, such that it outputs zero
        nn.init.xavier_uniform_(self.query_proj.weight)
        nn.init.xavier_uniform_(self.key_proj.weight)
        nn.init.zeros_(self.value_proj.weight)

    def init_cfg_encoder(self) -> Tuple[Callable[[torch.Tensor], torch.Tensor], int]:
        return (
            nn.Sequential(
                SinusoidalEncoding(self.hidden_dim),
                nn.Linear(self.hidden_dim, int(self.hidden_dim * self.mlp_ratio)),
                nn.ReLU(),
                nn.Linear(int(self.hidden_dim * self.mlp_ratio), self.hidden_dim),
                nn.LayerNorm(self.hidden_dim)
            ),
            self.hidden_dim,
        )

    def forward(self, x: torch.Tensor, cfg_scale: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: ({B}, T, input_dim)
            cfg_scale: ({B})
        
        Returns:
            x: ({B}, T, output_dim)
        """

        cfg_emb = self.cfg_encoder(cfg_scale)                   # ({B}, hidden_dim)

        query = self.query_proj(x)                              # ({B}, T, hidden_dim)
        key = self.key_proj(cfg_emb).unsqueeze(-2)              # ({B}, 1, hidden_dim)
        value = self.value_proj(cfg_emb).unsqueeze(-2)          # ({B}, 1, output_dim)

        return attention(query, key, value, scale=self.scale)


def attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, scale: float) -> torch.Tensor:
    """
    Args:
        q: ({B}, T, hidden_dim)
        k: ({B}, M, hidden_dim)
        v: ({B}, M, hidden_dim)
        scale: float

    Output: ({B}, T, hidden_dim)
    """

    q = q * scale
    attn = q @ k.transpose(-2, -1)      # ({B}, T, M)
    attn = torch.softmax(attn, dim=-1)  # ({B}, T, M)
    return attn @ v                     # ({B}, T, hidden_dim)


if __name__ == "__main__":
    B = 64
    T = 100
    I = 256
    D = 512

    print(f"variables: batch size={B}, input dim={I}, seq length={T}, hidden dim={D}")

    print("\ntesting sinusoidal encoding block...")

    sinenc = SinusoidalEncoding(D)
    pos = torch.linspace(0, 4, B)
    pos_enc = sinenc(pos)

    print(f"\tshape (expect [{B}, {D}]):", pos_enc.shape)

    print("\ntesting cfg adapter block...")

    cfg_adapter = CFGAdapterBlock(I, D)
    x = torch.randn(B, T, I)
    cfg_scale = torch.randn(B)

    print(f"\tinput mean: {x.mean().item():.3f}")
    print(f"\tinput std: {x.std().item():.3f}")

    out = cfg_adapter(x, cfg_scale)

    print(f"\tparameters: {sum(p.numel() for p in cfg_adapter.parameters() if p.requires_grad):,}")
    print(f"\tshape (expect [{B}, {T}, {I}]): {out.shape}")
    print(f"\toutput mean: {out.mean().item():.3f}")
    print(f"\toutput std: {out.std().item():.3f}")
