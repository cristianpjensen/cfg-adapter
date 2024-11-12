import torch
import torch.nn as nn
from diffusers import UNet2DConditionModel
from typing import Dict, List, Any, Optional, Union

from cfg_adapter import CFGAdapterBlock


class ImplicitCFGAdapterBlock(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

        self.adapter = CFGAdapterBlock(*args, **kwargs)
        self.cfg_scale = None

    def set_kwargs(self, kwargs: Dict[str, Any]):
        self.cfg_scale = kwargs["cfg_scale"]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.cfg_scale is None:
            raise ValueError("cfg_scale must be set before calling forward")

        adapter_input = x.reshape(*x.shape[:2], -1).transpose(1, 2)
        adapter_output = self.adapter(adapter_input, self.cfg_scale)
        adapter_output = adapter_output.transpose(1, 2).reshape(-1, self.adapter.output_dim, *x.shape[2:])
        self.cfg_scale = None

        return adapter_output


class BlockAdapter(nn.Module):
    def __init__(self, block, adapter, input_args: List[Union[str, int]]=[0]):
        super().__init__()
        self.input_args = input_args
        self.block = block
        self.adapter = adapter

    def forward(self, *args, **kwargs):
        adapter_inputs = []
        for arg in self.input_args:
            if isinstance(arg, str):
                adapter_inputs.append(kwargs[arg])
            else:
                adapter_inputs.append(args[arg])

        block_output = self.block(*args, **kwargs)
        adapter_output = self.adapter(*adapter_inputs)
        return block_output + adapter_output


class ModelWithAdapters(nn.Module):
    """Extracts injected adapters from model. This class is used to manage the adapters and the model
    together. The forward pass of the model is modified to include the adapter kwargs. The kwargs for
    the adapters should be given as the first argument---the rest of the arguments should be as given
    to the base model.

    Args:
        model (nn.Module): The model with injected adapters.
    
    Example:
        ```python
        import torch
        import torch.nn as nn
        from sd_ag import inject_adapters, ModelWithAdapters

        model = nn.Sequential(
            nn.Linear(32, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )
        # Adapters get the same input as the module they get attached to
        inject_adapters(model, {
            "0": nn.Linear(32, 128),
            "2": nn.Linear(128, 1),
        })
        model_with_adapters = ModelWithAdapters(model)
        model_with_adapters.train_adapters()

        x = torch.randn((4, 32))
        # If the adapters have a common keyword argument, it can be added to the dictionary in the
        # first argument
        y = model_with_adapters.forward(dict(), x)
        ```
    
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

    def set_kwargs(self, **kwargs):
        for adapter in self.adapters:
            if hasattr(adapter, "set_kwargs"):
                adapter.set_kwargs(kwargs)

    def train_adapters(self):
        self.model.requires_grad_(False)
        self.adapters.requires_grad_(True)

    def forward(self, adapter_kwargs, *args, **kwargs):
        self.set_kwargs(**adapter_kwargs)
        return self.model(*args, **kwargs)


def inject_adapters(
    model: nn.Module,
    adapters: Dict[str, nn.Module],
):
    for target, adapter in adapters.items():
        atoms = target.split(".")
        module = model.get_submodule(target)
        parent_module = model.get_submodule(".".join(atoms[:-1]))
        setattr(parent_module, atoms[-1], BlockAdapter(module, adapter))


def get_sd_ag_unet() -> ModelWithAdapters:
    unet = UNet2DConditionModel.from_pretrained("stabilityai/stable-diffusion-2-1", subfolder="unet")
    channels = unet.config.block_out_channels
    adapters = {
        "down_blocks.0.resnets.1": ImplicitCFGAdapterBlock(channels[0], 256),
        "down_blocks.1.resnets.1": ImplicitCFGAdapterBlock(channels[1], 256),
        "down_blocks.2.resnets.1": ImplicitCFGAdapterBlock(channels[2], 256),
        "down_blocks.3.resnets.1": ImplicitCFGAdapterBlock(channels[3], 256),
        "up_blocks.0.resnets.1": ImplicitCFGAdapterBlock(channels[3] * 2, 256, channels[3]),
        "up_blocks.1.resnets.1": ImplicitCFGAdapterBlock(channels[2] * 2, 256, channels[2]),
        "up_blocks.2.resnets.1": ImplicitCFGAdapterBlock(channels[1] * 2, 256, channels[1]),
        "up_blocks.3.resnets.1": ImplicitCFGAdapterBlock(channels[0] * 2, 256, channels[0]),
    }
    inject_adapters(unet, adapters)
    return ModelWithAdapters(unet)


if __name__ == "__main__":
    # Test 1
    print("Test 1")
    unet_with_adapters = get_sd_ag_unet()
    unet_with_adapters.train_adapters()
    print("  ✓ successfully injected adapters into model")

    sample = torch.randn(1, 4, 96, 96)
    encoder_hidden_states = torch.randn(1, 77, 1024)
    timestep = torch.tensor([0])

    unet_with_adapters.forward(
        dict(cfg_scale=torch.tensor([4.0])),
        sample, timestep, encoder_hidden_states,
    )
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
        "0": nn.Linear(32, 128),
        "2": nn.Linear(128, 1),
    })
    model_with_adapters = ModelWithAdapters(model)
    model_with_adapters.train_adapters()
    print("  ✓ successfully injected adapters into model")

    x = torch.randn((4, 32))
    # If the adapters have a common keyword argument, it can be added to the dictionary in the
    # first argument
    y = model_with_adapters.forward(dict(), x)
    print("  ✓ successfully performed a forward pass")
