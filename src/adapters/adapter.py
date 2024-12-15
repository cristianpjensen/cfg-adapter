import torch.nn as nn
from typing import Any


class ModelWithAdapters(nn.Module):
    """
    Handles adapters in the model, such as freezing the base model and setting keyword arguments for
    the adapters---external conditioning cannot be done another way, because the model is not "aware"
    there are adapters and we cannot change the internal `forward` of the model.

    Example:
    ```python
    # Input data, where `external_cond` is the external conditioning for the adapter, which the block
    # does not receive in the original model
    external_cond = torch.randn(4, 1280)
    x = torch.randn(4, 3, 256, 256)

    # `model` contains `Adapter` modules and we want to train the adapters with a frozen base model
    model = ModelWithAdapters(model)
    model.train_adapters()

    # Forward pass where we first have to pass external conditioning to the adapter
    model.set_adapter_kwargs(adapter_cond=external_cond)
    y = model(x)
    ```
    """

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    @property
    def adapters(self):
        adapters = []
        for module in self.model.modules():
            if isinstance(module, Adapter):
                adapters.append(module)

        return nn.ModuleList(adapters)

    @property
    def _blocks_with_adapters(self):
        """Blocks to which adapters have been added."""
        blocks = []
        for module in self.model.modules():
            if isinstance(module, Adapter):
                blocks.append(module.block)

        return nn.ModuleList(blocks)

    def adapter_parameters(self):
        """Parameters of all adapters."""

        for adapter in self.adapters:
            for param in adapter.adapter_parameters():
                yield param

    def named_adapter_parameters(self):
        """Named parameters of all adapters."""

        for adapter in self.adapters:
            for name, param in adapter.named_adapter_parameters():
                yield name, param

    def __getattr__(self, name):
        if name in ["model", "adapters", "_blocks_with_adapters", "adapter_parameters"]:
            return super().__getattr__(name)

        return getattr(self.model, name)

    def set_adapter_kwargs(self, **kwargs):
        for adapter in self.adapters:
            adapter.set_kwargs(**kwargs)

    def train_adapters(self):
        # It is important that it is done in this order
        self.model.requires_grad_(False)
        self.adapters.requires_grad_(True)
        self._blocks_with_adapters.requires_grad_(False)

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)


class Adapter(nn.Module):
    def __init__(self, block: nn.Module):
        super().__init__()
        self.block = block
        self.kwargs = None

    def adapter_parameters(self):
        """Parameters of adapter excluding block."""

        for _, p in self.named_adapter_parameters():
            yield p

    def named_adapter_parameters(self):
        """Named parameters of adapter excluding block."""

        for name, p in self.named_parameters():
            if name.split(".")[0] == "block":
                continue

            yield name, p

    def __getattr__(self, name):
        if name in ["block", "kwargs", "adapter_parameters"]:
            return super().__getattr__(name)

        # If it does not already exist in the adapter, try to get it from the block
        if hasattr(self.block, name):
            return self.block.__getattr__(name)

        return super().__getattr__(name)

    def set_kwargs(self, **kwargs):
        self.kwargs = kwargs

    def forward(self):
        raise NotImplementedError


class ExampleResidualLinearCondAdapter(Adapter):
    """Example of an adapter that makes use of external conditioning."""

    def __init__(self, block: nn.Module, in_dim: int, out_dim: int, cond_dim: int):
        super().__init__(block)
        self.linear1 = nn.Linear(in_dim, out_dim)
        self.linear2 = nn.Linear(cond_dim, out_dim)

    def forward(self, *args, **kwargs):
        if self.kwargs is None:
            raise ValueError(f"kwargs not set in {self.__class__.__name__}")

        if "adapter_cond" not in self.kwargs:
            raise ValueError(f"adapter_cond not in kwargs in {self.__class__.__name__}")

        adapter_cond = self.kwargs["adapter_cond"]
        block_out = self.block(*args, **kwargs)
        block_out[0] = block_out[0] + self.linear1(args[0]) + self.linear2(adapter_cond)
        return block_out
