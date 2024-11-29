import torch
import torch.nn as nn
from diffusers import UNet2DConditionModel
from typing import Dict, List, Any, Optional, Union, Tuple

from cfg_adapter import CFGAdapterBlock


class ImplicitCFGAdapterBlock(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

        self.adapter = CFGAdapterBlock(*args, **kwargs)
        self.cfg_scale = None
        self.encoder_hidden_states = None

    def set_kwargs(self, kwargs: Dict[str, Any]):
        self.cfg_scale = kwargs["cfg_scale"]
        self.encoder_hidden_states = kwargs["encoder_hidden_states"]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert self.cfg_scale is not None, "cfg_scale must be set before calling forward"
        assert self.encoder_hidden_states is not None, "encoder_hidden_states must be set before calling forward"

        return self.adapter(x, self.cfg_scale, self.encoder_hidden_states)


class BlockAdapter(nn.Module):
    def __init__(self, block, adapter, input_args: List[Union[str, int]]=[0], output_arg: Optional[Union[str, int]]=None):
        super().__init__()
        self.input_args = input_args
        self.output_arg = output_arg
        self.block = block
        self.adapter = adapter

    def forward(self, *args, **kwargs):
        adapter_inputs = []
        for arg in self.input_args:
            if isinstance(arg, str):
                adapter_inputs.append(kwargs[arg])
            else:
                adapter_inputs.append(args[arg])

        output = self.block(*args, **kwargs)
        adapter_output = self.adapter(*adapter_inputs)

        if self.output_arg is None:
            output += adapter_output
        else:
            if isinstance(output, tuple):
                output = list(output)
                output[self.output_arg] += adapter_output
                output = tuple(output)
            else:
                output[self.output_arg] += adapter_output

        return output


class ModelWithAdapters(nn.Module):
    """Extracts injected adapters from model. This class is used to manage the adapters and the model
    together. The forward pass of the model is modified to include the adapter kwargs. The kwargs for
    the adapters should be given as the first argument---the rest of the arguments should be as given
    to the base model.

    Args:
        model (nn.Module): The model with injected adapters.
    """

    def __init__(self, model: nn.Module):
        super().__init__()

        # Fetch adapters in the model and save them
        adapters = []
        for module in model.modules():
            if isinstance(module, BlockAdapter):
                adapters.append(module.adapter)

        self.model = model
        self.adapters = nn.ModuleList(adapters)

    def __getattr__(self, name):
        if name in ["model", "adapters"]:
            return super().__getattr__(name)

        return getattr(self.model, name)

    def set_kwargs(self, **kwargs):
        for adapter in self.adapters:
            if hasattr(adapter, "set_kwargs"):
                adapter.set_kwargs(kwargs)

    def train_adapters(self):
        self.model.requires_grad_(False)
        self.adapters.requires_grad_(True)

    def forward_with_adapter_kwargs(self, adapter_kwargs, *args, **kwargs):
        self.set_kwargs(**adapter_kwargs)
        return self.model(*args, **kwargs)

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)


def inject_adapters(
    model: nn.Module,
    adapters: Dict[str, Union[nn.Module, Tuple[nn.Module, Dict]]],
):
    for target, adapter in adapters.items():
        if isinstance(adapter, tuple):
            adapter, kwargs = adapter
        else:
            kwargs = dict()

        atoms = target.split(".")
        module = model.get_submodule(target)
        parent_module = model.get_submodule(".".join(atoms[:-1]))
        setattr(parent_module, atoms[-1], BlockAdapter(module, adapter, **kwargs))


def get_sd_ag_unet() -> ModelWithAdapters:
    unet = UNet2DConditionModel.from_pretrained("stabilityai/stable-diffusion-2-1", subfolder="unet")
    channels = unet.config.block_out_channels
    num_heads = unet.config.attention_head_dim
    cond_dim = unet.config.cross_attention_dim
    hidden_dim = 320
    adapters = {
        "down_blocks.0.attentions.0.transformer_blocks.0.attn2": ImplicitCFGAdapterBlock(channels[0], hidden_dim, cond_dim=cond_dim, num_heads=num_heads[0]),
        "down_blocks.0.attentions.1.transformer_blocks.0.attn2": ImplicitCFGAdapterBlock(channels[0], hidden_dim, cond_dim=cond_dim, num_heads=num_heads[0]),
        "down_blocks.1.attentions.0.transformer_blocks.0.attn2": ImplicitCFGAdapterBlock(channels[1], hidden_dim, cond_dim=cond_dim, num_heads=num_heads[1]),
        "down_blocks.1.attentions.1.transformer_blocks.0.attn2": ImplicitCFGAdapterBlock(channels[1], hidden_dim, cond_dim=cond_dim, num_heads=num_heads[1]),
        "down_blocks.2.attentions.0.transformer_blocks.0.attn2": ImplicitCFGAdapterBlock(channels[2], hidden_dim, cond_dim=cond_dim, num_heads=num_heads[2]),
        "down_blocks.2.attentions.1.transformer_blocks.0.attn2": ImplicitCFGAdapterBlock(channels[2], hidden_dim, cond_dim=cond_dim, num_heads=num_heads[2]),
        "mid_block.attentions.0.transformer_blocks.0.attn2": ImplicitCFGAdapterBlock(channels[3], hidden_dim, cond_dim=cond_dim, num_heads=num_heads[3]),
        "up_blocks.1.attentions.0.transformer_blocks.0.attn2": ImplicitCFGAdapterBlock(channels[2], hidden_dim, cond_dim=cond_dim, num_heads=num_heads[2]),
        "up_blocks.1.attentions.1.transformer_blocks.0.attn2": ImplicitCFGAdapterBlock(channels[2], hidden_dim, cond_dim=cond_dim, num_heads=num_heads[2]),
        "up_blocks.1.attentions.2.transformer_blocks.0.attn2": ImplicitCFGAdapterBlock(channels[2], hidden_dim, cond_dim=cond_dim, num_heads=num_heads[2]),
        "up_blocks.2.attentions.0.transformer_blocks.0.attn2": ImplicitCFGAdapterBlock(channels[1], hidden_dim, cond_dim=cond_dim, num_heads=num_heads[1]),
        "up_blocks.2.attentions.1.transformer_blocks.0.attn2": ImplicitCFGAdapterBlock(channels[1], hidden_dim, cond_dim=cond_dim, num_heads=num_heads[1]),
        "up_blocks.2.attentions.2.transformer_blocks.0.attn2": ImplicitCFGAdapterBlock(channels[1], hidden_dim, cond_dim=cond_dim, num_heads=num_heads[1]),
        "up_blocks.3.attentions.0.transformer_blocks.0.attn2": ImplicitCFGAdapterBlock(channels[0], hidden_dim, cond_dim=cond_dim, num_heads=num_heads[0]),
        "up_blocks.3.attentions.1.transformer_blocks.0.attn2": ImplicitCFGAdapterBlock(channels[0], hidden_dim, cond_dim=cond_dim, num_heads=num_heads[0]),
        "up_blocks.3.attentions.2.transformer_blocks.0.attn2": ImplicitCFGAdapterBlock(channels[0], hidden_dim, cond_dim=cond_dim, num_heads=num_heads[0]),
    }
    inject_adapters(unet, adapters)
    return ModelWithAdapters(unet)


def get_sdxl_ag_unet() -> ModelWithAdapters:
    unet = UNet2DConditionModel.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", subfolder="unet")
    channels = unet.config.block_out_channels
    cond_dim = unet.config.cross_attention_dim
    hidden_dim = 512
    adapters = {
        "down_blocks.1.attentions.0": (ImplicitCFGAdapterBlock(channels[1], hidden_dim, cond_dim), dict(output_arg=0)),
        "down_blocks.1.attentions.1": (ImplicitCFGAdapterBlock(channels[1], hidden_dim, cond_dim), dict(output_arg=0)),
        "down_blocks.2.attentions.0": (ImplicitCFGAdapterBlock(channels[2], hidden_dim, cond_dim), dict(output_arg=0)),
        "down_blocks.2.attentions.1": (ImplicitCFGAdapterBlock(channels[2], hidden_dim, cond_dim), dict(output_arg=0)),
        "mid_block.attentions.0": (ImplicitCFGAdapterBlock(channels[2], hidden_dim, cond_dim), dict(output_arg=0)),
        "up_blocks.0.attentions.0": (ImplicitCFGAdapterBlock(channels[2], hidden_dim, cond_dim), dict(output_arg=0)),
        "up_blocks.0.attentions.1": (ImplicitCFGAdapterBlock(channels[2], hidden_dim, cond_dim), dict(output_arg=0)),
        "up_blocks.0.attentions.2": (ImplicitCFGAdapterBlock(channels[2], hidden_dim, cond_dim), dict(output_arg=0)),
        "up_blocks.1.attentions.0": (ImplicitCFGAdapterBlock(channels[1], hidden_dim, cond_dim), dict(output_arg=0)),
        "up_blocks.1.attentions.1": (ImplicitCFGAdapterBlock(channels[1], hidden_dim, cond_dim), dict(output_arg=0)),
        "up_blocks.1.attentions.2": (ImplicitCFGAdapterBlock(channels[1], hidden_dim, cond_dim), dict(output_arg=0)),
    }
    inject_adapters(unet, adapters)
    return ModelWithAdapters(unet)


GET_ADAPTER_UNET = {
    "stabilityai/stable-diffusion-2-1": get_sd_ag_unet,
    "stabilityai/stable-diffusion-xl-base-1.0": get_sdxl_ag_unet,
}


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Test 1
    print("Test 1")
    unet_with_adapters = get_sd_ag_unet().to(device)
    unet_with_adapters.train_adapters()
    print(f"  number of parameters: {unet_with_adapters.num_parameters(only_trainable=True):,}")
    print("  ✓ successfully injected adapters into model")

    sample = torch.randn(1, 4, 96, 96).to(device)
    encoder_hidden_states = torch.randn(1, 77, 1024).to(device)
    timestep = torch.tensor([0]).to(device)

    unet_with_adapters.set_kwargs(cfg_scale=torch.tensor([4.0], device=device))
    unet_with_adapters.forward(sample, timestep, encoder_hidden_states)
    print("  ✓ successfully performed a forward pass")

    # Test 2
    print("Test 2")
    model = nn.Sequential(
        nn.Linear(32, 128),
        nn.ReLU(),
        nn.Linear(128, 1),
    )
    # Adapters get the same input as the module they get attached to
    inject_adapters(model, {
        "0": (nn.Linear(32, 128), dict()),
        "2": (nn.Linear(128, 1), dict()),
    })
    model_with_adapters = ModelWithAdapters(model).to(device)
    model_with_adapters.train_adapters()
    print("  ✓ successfully injected adapters into model")

    x = torch.randn((4, 32)).to(device)
    # If the adapters have a common keyword argument, it can be added to the dictionary in the
    # first argument
    y = model_with_adapters.forward(x)
    print("  ✓ successfully performed a forward pass")
