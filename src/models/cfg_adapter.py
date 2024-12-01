import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from diffusers.models.attention_processor import Attention
import math
from typing import Optional

from src.adapters import Adapter


class CrossAttentionCFGAdapter(Adapter):
    """Adapter that conditions on the CFG scale."""

    def __init__(
        self,
        block: Attention,
        input_dim: int,
        hidden_dim: int,
        prompt_dim: Optional[int] = None,
        use_prompt_cond: bool=False,
        use_neg_prompt_cond: bool=False,
        use_block_query: bool=False,
        num_heads: int=1,
        cfg_mlp_ratio: float=4.0,
    ):
        super().__init__(block)

        self.num_heads = num_heads
        self.use_prompt_cond = use_prompt_cond
        self.use_neg_prompt_cond = use_neg_prompt_cond

        # CFG scale encoder
        cfg_dim = hidden_dim
        self.cfg_enc = ScalarEncoder(cfg_dim, cfg_mlp_ratio)

        # Query map
        if use_block_query:
            self.q_proj = self.block.to_q
            hidden_dim = self.q_proj.out_features
        else:
            self.q_proj = nn.Linear(input_dim, hidden_dim, bias=False)
            nn.init.xavier_normal_(self.q_proj.weight)

        # Key and value maps for CFG scale
        self.kv_proj = nn.Linear(cfg_dim, 2 * hidden_dim, bias=False)
        nn.init.xavier_normal_(self.kv_proj.weight)
        nn.init.zeros_(self.kv_proj.weight[hidden_dim:])

        # Key and value maps for prompt
        if use_prompt_cond:
            assert prompt_dim is not None, "prompt_dim must be set if using prompt condition"
            self.prompt_kv_proj = nn.Linear(prompt_dim, 2 * hidden_dim, bias=False)
            nn.init.xavier_normal_(self.prompt_kv_proj.weight)
            nn.init.zeros_(self.prompt_kv_proj.weight[hidden_dim:])

        # Key and value maps for negative prompt
        if use_neg_prompt_cond:
            assert prompt_dim is not None, "prompt_dim must be set if using neg prompt condition"
            self.neg_prompt_kv_proj = nn.Linear(prompt_dim, 2 * hidden_dim, bias=False)
            nn.init.xavier_normal_(self.neg_prompt_kv_proj.weight)
            nn.init.zeros_(self.neg_prompt_kv_proj.weight[hidden_dim:])

        self.out_proj = nn.Linear(hidden_dim, input_dim, bias=False)
        nn.init.xavier_normal_(self.out_proj.weight)

        self.scale = 1.0 / math.sqrt(hidden_dim)

    def _adapter_forward(
        self,
        x: torch.Tensor,
        cfg_scale: torch.Tensor,
        prompt_cond: Optional[torch.Tensor]=None,
        neg_prompt_cond: Optional[torch.Tensor]=None,
    ):
        cfg_emb = self.cfg_enc(cfg_scale).unsqueeze(-2)                  # (...B, 1, hidden_dim)

        q = self.q_proj(x)                                               # (...B, T, hidden_dim)
        kv = self.kv_proj(cfg_emb)                                       # (...B, 1, hidden_dim * 2)
        k, v = kv.chunk(2, dim=-1)                                       # (...B, 1, hidden_dim)

        if self.use_prompt_cond:
            prompt_kv = self.prompt_kv_proj(prompt_cond)                 # (...B, S, hidden_dim * 2)
            prompt_k, prompt_v = prompt_kv.chunk(2, dim=-1)              # (...B, S, hidden_dim)
            k = torch.cat([k, prompt_k], dim=-2)                         # (...B, 1 + S, hidden_dim)
            v = torch.cat([v, prompt_v], dim=-2)                         # (...B, 1 + S, hidden_dim)

        if self.use_neg_prompt_cond:
            neg_prompt_kv = self.neg_prompt_kv_proj(neg_prompt_cond)     # (...B, S, hidden_dim * 2)
            neg_prompt_k, neg_prompt_v = neg_prompt_kv.chunk(2, dim=-1)  # (...B, S, hidden_dim)
            k = torch.cat([k, neg_prompt_k], dim=-2)                     # (...B, 1 + S + S, hidden_dim)
            v = torch.cat([v, neg_prompt_v], dim=-2)                     # (...B, 1 + S + S, hidden_dim)
        
        def compute_cross_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
            """
            Args:
                q: (...B, T, hidden_dim)
                k: (...B, L, hidden_dim)
                v: (...B, L, hidden_dim)

            Returns: (...B, T, hidden_dim)
            """

            # Apply multi-head attention
            q = q.view(*q.shape[:-1], self.num_heads, -1).transpose(-3, -2)  # (...B, num_heads, T, head_dim)
            k = k.view(*k.shape[:-1], self.num_heads, -1).transpose(-3, -2)  # (...B, num_heads, L, head_dim)
            v = v.view(*v.shape[:-1], self.num_heads, -1).transpose(-3, -2)  # (...B, num_heads, L, head_dim)

            out = F.scaled_dot_product_attention(q, k, v, scale=self.scale)  # (...B, num_heads, T, head_dim)
            out = out.transpose(-3, -2)                                      # (...B, T, num_heads, head_dim)
            out = out.reshape(*out.shape[:-2], -1)                           # (...B, T, hidden_dim)
            return self.out_proj(out)

        return checkpoint(compute_cross_attention, q, k, v, use_reentrant=True)


    def forward(self, *args, **kwargs):
        """
        Args:
            x: (...B, T, hidden_dim)
        
        Returns: (...B, T, hidden_dim)
        """

        assert self.kwargs is not None, f"kwargs must be set in {self.__class__.__name__}"
        assert "cfg_scale" in self.kwargs, f"cfg_scale must be set in kwargs in {self.__class__.__name__}"
        assert not (self.use_prompt_cond and "prompt" not in self.kwargs), f"prompt not in kwargs in {self.__class__.__name__}"
        assert not (self.use_neg_prompt_cond and "neg_prompt" not in self.kwargs), f"neg_prompt not in kwargs in {self.__class__.__name__}"

        x = args[0]
        cfg_scale = self.kwargs["cfg_scale"]
        prompt_cond = self.kwargs.get("prompt", None)
        neg_prompt_cond = self.kwargs.get("neg_prompt", None)

        return self.block(*args, **kwargs) + self._adapter_forward(x, cfg_scale, prompt_cond, neg_prompt_cond)


class ScalarEncoder(nn.Module):
    def __init__(self, enc_dim: int, mlp_ratio: float=4.0):
        super().__init__()

        hidden_dim = int(enc_dim * mlp_ratio)
        self.net = nn.Sequential(
            SinusoidalEncoding(enc_dim),
            nn.Linear(enc_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, enc_dim),
            nn.LayerNorm(enc_dim),
        )

    def forward(self, cfg_scale: torch.Tensor) -> torch.Tensor:
        return self.net(cfg_scale)


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

        pos_div = pos.float().unsqueeze(-1) * self.div_term
        return torch.cat([torch.sin(pos_div), torch.cos(pos_div)], dim=-1)
