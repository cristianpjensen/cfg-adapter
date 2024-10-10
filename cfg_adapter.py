import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Callable, Tuple


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
    def __init__(self, input_dim: int, hidden_dim: int, mlp_ratio: int=4):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.mlp_ratio = mlp_ratio
        self.cfg_encoder, cfg_dim = self.init_cfg_encoder()

        assert type(cfg_dim) == int, "cfg_dim must be an integer"
        assert cfg_dim > 0, "cfg_dim must be a positive integer"

        self.query_proj = nn.Linear(input_dim, hidden_dim, bias=False)
        self.key_proj = nn.Linear(cfg_dim, hidden_dim, bias=False)
        self.value_proj = nn.Linear(cfg_dim, input_dim, bias=False)

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
            x: ({B}, T, input_dim)
        """

        cfg_emb = self.cfg_encoder(cfg_scale)                   # ({B}, hidden_dim)

        query = self.query_proj(x)                              # ({B}, T, hidden_dim)
        key = self.key_proj(cfg_emb).unsqueeze(-2)              # ({B}, 1, hidden_dim)
        value = self.value_proj(cfg_emb).unsqueeze(-2)          # ({B}, 1, hidden_dim)

        return F.scaled_dot_product_attention(query, key, value, scale=self.hidden_dim)


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
    out = cfg_adapter(x, cfg_scale)

    print(f"\tparameters:", sum(p.numel() for p in cfg_adapter.parameters() if p.requires_grad))
    print(f"\tshape (expect [{B}, {T}, {I}]):", out.shape)
    print("\tmean:", out.mean().item())
    print("\tstd:", out.std().item())
