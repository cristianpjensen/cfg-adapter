import torch
import torch.nn as nn

from DiT.models import DiT
from cfg_adapter import CFGAdapterBlock


class DiT_AG(DiT):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        mlp_ratio = kwargs.get("mlp_ratio", 4)
        hidden_dim = kwargs.get("hidden_size", 1152)

        # Freeze weights of base model such that only adapter weights are updated (and considered by
        # the optimizer)
        for param in self.parameters():
            param.requires_grad = False

        # Define adapters
        self.adapters = nn.ModuleList([
            CFGAdapterBlock(
                input_dim=hidden_dim,
                hidden_dim=256,
                mlp_ratio=mlp_ratio,
            ) for _ in range(len(self.blocks))
        ])

    def forward(self, x, t, y, cfg_scale):
        """
        Forward pass of DiT with CFG encoder.
        x: (N, C, H, W) tensor of spatial inputs (images or latent representations of images)
        t: (N,) tensor of diffusion timesteps
        y: (N,) tensor of class labels
        cfg_scale: (N,) tensor of CFG scales
        """
        x = self.x_embedder(x) + self.pos_embed  # (N, T, D), where T = H * W / patch_size ** 2
        t = self.t_embedder(t)                   # (N, D)
        y = self.y_embedder(y, self.training)    # (N, D)
        c = t + y                                # (N, D)

        for block, adapter in zip(self.blocks, self.adapters):
            x_block = block(x, c)                # (N, T, D)
            x_adapt = adapter(x, cfg_scale)      # (N, T, D)
            x = x_block + x_adapt                # (N, T, D)  

        x = self.final_layer(x, c)               # (N, T, patch_size ** 2 * out_channels)
        x = self.unpatchify(x)                   # (N, out_channels, H, W)

        return x

    def forward_with_cfg(self, x, t, y, cfg_scale):
        return self.forward(x, t, y, cfg_scale)


if __name__ == "__main__":
    print("testing DiT with CFG adapter...")

    B = 2
    C = 4
    I = 32
    H = 1152
    cfg_dit = DiT_AG(input_size=I, in_channels=C, hidden_size=H)
    cfg_dit = cfg_dit.train()

    x = torch.randn(B, C, I, I)
    t = torch.rand(B)
    y = torch.randint(0, 1000, (B,))
    cfg_scale = torch.randn(B)
    y = cfg_dit(x, t, y, cfg_scale)

    print(f"\tshape (expect [{B}, {C*2}, {I}, {I}]):", y.shape)
    print(f"\tparameters:", sum(p.numel() for p in cfg_dit.parameters()))
    print(f"\tadapter parameters:", sum(p.numel() for p in cfg_dit.adapter_parameters()))
