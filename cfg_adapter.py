import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
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
            pos: (...B)
        
        Returns: (...B, hidden_dim)
        """

        pos_div = torch.einsum("...b, d -> ...bd", pos.float(), self.div_term)
        return torch.cat([torch.sin(pos_div), torch.cos(pos_div)], dim=-1)


class CFGAdapterBlock(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        cond_dim: Optional[int]=None,
        num_heads: int=1,
        output_dim: Optional[int]=None,
        mlp_ratio: float=4.0,
    ):
        super().__init__()

        output_dim = output_dim if output_dim is not None else input_dim

        self.cfg_encoder, cfg_dim = self.init_cfg_encoder(hidden_dim, mlp_ratio)
        self.query_proj = nn.Linear(input_dim, hidden_dim, bias=False)
        self.kv_proj = nn.Linear(cfg_dim, hidden_dim + output_dim, bias=False)
        self.cond_kv_proj = nn.Linear(cond_dim, hidden_dim + output_dim, bias=False) if cond_dim is not None else None

        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.scale = 1.0 / math.sqrt(hidden_dim)

        # Zero-initialize value projection, such that it does not change the network at initialization
        nn.init.xavier_normal_(self.query_proj.weight)
        nn.init.xavier_normal_(self.kv_proj.weight)
        nn.init.zeros_(self.kv_proj.weight[hidden_dim:])

        if self.cond_kv_proj is not None:
            nn.init.xavier_uniform_(self.cond_kv_proj.weight)
            nn.init.zeros_(self.cond_kv_proj.weight[hidden_dim:])

    def init_cfg_encoder(self, hidden_dim: int, mlp_ratio: float) -> Tuple[Callable[[torch.Tensor], torch.Tensor], int]:
        return (
            nn.Sequential(
                SinusoidalEncoding(hidden_dim),
                nn.Linear(hidden_dim, int(hidden_dim * mlp_ratio)),
                nn.ReLU(),
                nn.Linear(int(hidden_dim * mlp_ratio), hidden_dim),
                nn.LayerNorm(hidden_dim)
            ),
            hidden_dim,
        )

    def forward(self, x: torch.Tensor, cfg_scale: torch.Tensor, cond_emb: Optional[torch.Tensor]=None) -> torch.Tensor:
        """
        Args:
            x: (...B, T, input_dim)
            cfg_scale: (...B)
            cond_emb: (...B, S, cond_dim) or None
        
        Returns:
            x: (...B, T, output_dim)
        """

        cfg_emb = self.cfg_encoder(cfg_scale).unsqueeze(-2)                                     # (...B, 1, hidden_dim)

        # Compute query, key, and value for cross-attention
        query = self.query_proj(x)                                                              # (...B, T, hidden_dim)
        kv = self.kv_proj(cfg_emb)                                                              # (...B, 1, hidden_dim + output_dim)
        key, value = kv.split([self.hidden_dim, self.output_dim], dim=-1)

        # Add conditioning by concatenating key and value with conditioned key and value
        if cond_emb is not None and self.cond_kv_proj is not None:
            cond_kv = self.cond_kv_proj(cond_emb)                                               # (...B, S, hidden_dim + output_dim)
            cond_key, cond_value = cond_kv.split([self.hidden_dim, self.output_dim], dim=-1)
            key = torch.cat([key, cond_key], dim=-2)                                            # (...B, S + 1, hidden_dim)
            value = torch.cat([value, cond_value], dim=-2)                                      # (...B, S + 1, output_dim)

        def compute_cross_attention(query, key, value):
            # Apply multi-head attention
            query = query.view(*query.shape[:-1], self.num_heads, -1).transpose(-3, -2)             # (...B, num_heads, T, hidden_dim)
            key = key.view(*key.shape[:-1], self.num_heads, -1).transpose(-3, -2)                   # (...B, num_heads, L, hidden_dim)
            value = value.view(*value.shape[:-1], self.num_heads, -1).transpose(-3, -2)             # (...B, num_heads, L, output_dim)

            out = F.scaled_dot_product_attention(query, key, value, scale=self.scale)               # (...B, num_heads, T, output_dim)
            out = out.transpose(-3, -2)                                                             # (...B, T, num_heads, output_dim)
            out = out.reshape(*out.shape[:-2], -1)                                                  # (...B, T, output_dim)

            return out

        return checkpoint(compute_cross_attention, query, key, value, use_reentrant=True)


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

    cfg_adapter = CFGAdapterBlock(I, D, num_heads=4)
    x = torch.randn(B, T, I)
    cfg_scale = torch.randn(B)

    print(f"\tinput mean: {x.mean().item():.3f}")
    print(f"\tinput std: {x.std().item():.3f}")

    out = cfg_adapter(x, cfg_scale)

    print(f"\tparameters: {sum(p.numel() for p in cfg_adapter.parameters() if p.requires_grad):,}")
    print(f"\tshape (expect [{B}, {T}, {I}]): {out.shape}")
    print(f"\toutput mean: {out.mean().item():.3f}")
    print(f"\toutput std: {out.std().item():.3f}")
