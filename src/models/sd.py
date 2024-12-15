from src.adapters.inject_adapters import inject_adapters
from src.models.cfg_adapter import CrossAttentionCFGAdapter
from src.adapters import AdapterConfig, ModelWithAdapters
from diffusers import UNet2DConditionModel


def get_sd_adapter_unet(
    base_model: str,
    hidden_dim=320,
    use_prompt_cond=False,
    use_neg_prompt_cond=False,
    use_block_query=False,
) -> ModelWithAdapters:
    unet = UNet2DConditionModel.from_pretrained(base_model, subfolder="unet")

    channels = unet.config.block_out_channels
    layers_per_block = unet.config.layers_per_block
    transformer_layers = unet.config.transformer_layers_per_block

    if isinstance(transformer_layers, int):
        transformer_layers = [transformer_layers] * len(channels)

    down_block_types = unet.config.down_block_types
    up_block_types = unet.config.up_block_types

    num_heads = unet.config.attention_head_dim
    prompt_dim = unet.config.cross_attention_dim

    adapters = {}

    # Down blocks
    for i, block_type in enumerate(down_block_types):
        if block_type != "CrossAttnDownBlock2D":
            continue

        for j in range(layers_per_block):
            for k in range(transformer_layers[i]):
                adapters[f"down_blocks.{i}.attentions.{j}.transformer_blocks.{k}.attn2"] = AdapterConfig(
                    adapter=CrossAttentionCFGAdapter,
                    kwargs=dict(
                        input_dim=channels[i],
                        hidden_dim=hidden_dim,
                        prompt_dim=prompt_dim,
                        use_prompt_cond=use_prompt_cond,
                        use_neg_prompt_cond=use_neg_prompt_cond,
                        use_block_query=use_block_query,
                        num_heads=num_heads[i],
                    )
                )

    # Mid block
    for k in range(transformer_layers[-1]):
        adapters[f"mid_block.attentions.0.transformer_blocks.{k}.attn2"] = AdapterConfig(
            adapter=CrossAttentionCFGAdapter,
            kwargs=dict(
                input_dim=channels[-1],
                hidden_dim=hidden_dim,
                prompt_dim=prompt_dim,
                use_prompt_cond=use_prompt_cond,
                use_neg_prompt_cond=use_neg_prompt_cond,
                use_block_query=use_block_query,
                num_heads=num_heads[-1],
            )
        )

    # Up blocks
    for i, block_type in enumerate(up_block_types):
        if block_type != "CrossAttnUpBlock2D":
            continue

        for j in range(layers_per_block):
            for k in range(transformer_layers[::-1][i]):
                adapters[f"up_blocks.{i}.attentions.{j}.transformer_blocks.{k}.attn2"] = AdapterConfig(
                    adapter=CrossAttentionCFGAdapter,
                    kwargs=dict(
                        input_dim=channels[::-1][i],
                        hidden_dim=hidden_dim,
                        prompt_dim=prompt_dim,
                        use_prompt_cond=use_prompt_cond,
                        use_neg_prompt_cond=use_neg_prompt_cond,
                        use_block_query=use_block_query,
                        num_heads=num_heads[i],
                    )
                )

    return inject_adapters(unet, adapters)
