from src.adapters.inject_adapters import inject_adapters
from src.models.cfg_adapter import CrossAttentionCFGAdapter
from src.adapters import AdapterConfig, ModelWithAdapters
from diffusers import Transformer2DModel


def get_dit_adapter_transformer(hidden_dim=320, image_size: int=256) -> ModelWithAdapters:
    assert image_size in [256, 512], "only 256 and 512 image sizes are supported by DiT"

    transformer = Transformer2DModel.from_pretrained(f"facebook/DiT-XL-2-{image_size}", subfolder="transformer")

    head_dim = transformer.config.attention_head_dim
    num_heads = transformer.config.num_attention_heads
    input_dim = num_heads * head_dim

    adapters = {}
    for i in range(transformer.config.num_layers):
        adapters[f"transformer_blocks.{i}.attn1"] = AdapterConfig(
            adapter=CrossAttentionCFGAdapter,
            kwargs=dict(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                num_heads=num_heads,
            )
        )

    return inject_adapters(transformer, adapters)
