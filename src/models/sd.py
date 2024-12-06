from src.adapters.inject_adapters import inject_adapters
from src.models.cfg_adapter import CrossAttentionCFGAdapter
from src.adapters import AdapterConfig, ModelWithAdapters
from diffusers import UNet2DConditionModel


def get_sd_adapter_unet(
    hidden_dim=320,
    use_prompt_cond=False,
    use_neg_prompt_cond=False,
    use_block_query=False,
) -> ModelWithAdapters:
    unet = UNet2DConditionModel.from_pretrained("stabilityai/stable-diffusion-2-1", subfolder="unet")
    channels = unet.config.block_out_channels
    num_heads = unet.config.attention_head_dim
    prompt_dim = unet.config.cross_attention_dim

    adapters = {
        "down_blocks.0.attentions.0.transformer_blocks.0.attn2": AdapterConfig(
            adapter=CrossAttentionCFGAdapter,
            kwargs=dict(
                input_dim=channels[0],
                hidden_dim=hidden_dim,
                prompt_dim=prompt_dim,
                use_prompt_cond=use_prompt_cond,
                use_neg_prompt_cond=use_neg_prompt_cond,
                use_block_query=use_block_query,
                num_heads=num_heads[0],
            )
        ),
        "down_blocks.0.attentions.1.transformer_blocks.0.attn2": AdapterConfig(
            adapter=CrossAttentionCFGAdapter,
            kwargs=dict(
                input_dim=channels[0],
                hidden_dim=hidden_dim,
                prompt_dim=prompt_dim,
                use_prompt_cond=use_prompt_cond,
                use_neg_prompt_cond=use_neg_prompt_cond,
                use_block_query=use_block_query,
                num_heads=num_heads[0],
            )
        ),
        "down_blocks.1.attentions.0.transformer_blocks.0.attn2": AdapterConfig(
            adapter=CrossAttentionCFGAdapter,
            kwargs=dict(
                input_dim=channels[1],
                hidden_dim=hidden_dim,
                prompt_dim=prompt_dim,
                use_prompt_cond=use_prompt_cond,
                use_neg_prompt_cond=use_neg_prompt_cond,
                use_block_query=use_block_query,
                num_heads=num_heads[1],
            )
        ),
        "down_blocks.1.attentions.1.transformer_blocks.0.attn2": AdapterConfig(
            adapter=CrossAttentionCFGAdapter,
            kwargs=dict(
                input_dim=channels[1],
                hidden_dim=hidden_dim,
                prompt_dim=prompt_dim,
                use_prompt_cond=use_prompt_cond,
                use_neg_prompt_cond=use_neg_prompt_cond,
                use_block_query=use_block_query,
                num_heads=num_heads[1],
            )
        ),
        "down_blocks.2.attentions.0.transformer_blocks.0.attn2": AdapterConfig(
            adapter=CrossAttentionCFGAdapter,
            kwargs=dict(
                input_dim=channels[2],
                hidden_dim=hidden_dim,
                prompt_dim=prompt_dim,
                use_prompt_cond=use_prompt_cond,
                use_neg_prompt_cond=use_neg_prompt_cond,
                use_block_query=use_block_query,
                num_heads=num_heads[2],
            )
        ),
        "down_blocks.2.attentions.1.transformer_blocks.0.attn2": AdapterConfig(
            adapter=CrossAttentionCFGAdapter,
            kwargs=dict(
                input_dim=channels[2],
                hidden_dim=hidden_dim,
                prompt_dim=prompt_dim,
                use_prompt_cond=use_prompt_cond,
                use_neg_prompt_cond=use_neg_prompt_cond,
                use_block_query=use_block_query,
                num_heads=num_heads[2],
            )
        ),
        "mid_block.attentions.0.transformer_blocks.0.attn2": AdapterConfig(
            adapter=CrossAttentionCFGAdapter,
            kwargs=dict(
                input_dim=channels[3],
                hidden_dim=hidden_dim,
                prompt_dim=prompt_dim,
                use_prompt_cond=use_prompt_cond,
                use_neg_prompt_cond=use_neg_prompt_cond,
                use_block_query=use_block_query,
                num_heads=num_heads[3],
            )
        ),
        "up_blocks.1.attentions.0.transformer_blocks.0.attn2": AdapterConfig(
            adapter=CrossAttentionCFGAdapter,
            kwargs=dict(
                input_dim=channels[2],
                hidden_dim=hidden_dim,
                prompt_dim=prompt_dim,
                use_prompt_cond=use_prompt_cond,
                use_neg_prompt_cond=use_neg_prompt_cond,
                use_block_query=use_block_query,
                num_heads=num_heads[2],
            )
        ),
        "up_blocks.1.attentions.1.transformer_blocks.0.attn2": AdapterConfig(
            adapter=CrossAttentionCFGAdapter,
            kwargs=dict(
                input_dim=channels[2],
                hidden_dim=hidden_dim,
                prompt_dim=prompt_dim,
                use_prompt_cond=use_prompt_cond,
                use_neg_prompt_cond=use_neg_prompt_cond,
                use_block_query=use_block_query,
                num_heads=num_heads[2],
            )
        ),
        "up_blocks.1.attentions.2.transformer_blocks.0.attn2": AdapterConfig(
            adapter=CrossAttentionCFGAdapter,
            kwargs=dict(
                input_dim=channels[2],
                hidden_dim=hidden_dim,
                prompt_dim=prompt_dim,
                use_prompt_cond=use_prompt_cond,
                use_neg_prompt_cond=use_neg_prompt_cond,
                use_block_query=use_block_query,
                num_heads=num_heads[2],
            )
        ),
        "up_blocks.2.attentions.0.transformer_blocks.0.attn2": AdapterConfig(
            adapter=CrossAttentionCFGAdapter,
            kwargs=dict(
                input_dim=channels[1],
                hidden_dim=hidden_dim,
                prompt_dim=prompt_dim,
                use_prompt_cond=use_prompt_cond,
                use_neg_prompt_cond=use_neg_prompt_cond,
                use_block_query=use_block_query,
                num_heads=num_heads[1],
            )
        ),
        "up_blocks.2.attentions.1.transformer_blocks.0.attn2": AdapterConfig(
            adapter=CrossAttentionCFGAdapter,
            kwargs=dict(
                input_dim=channels[1],
                hidden_dim=hidden_dim,
                prompt_dim=prompt_dim,
                use_prompt_cond=use_prompt_cond,
                use_neg_prompt_cond=use_neg_prompt_cond,
                use_block_query=use_block_query,
                num_heads=num_heads[1],
            )
        ),
        "up_blocks.2.attentions.2.transformer_blocks.0.attn2": AdapterConfig(
            adapter=CrossAttentionCFGAdapter,
            kwargs=dict(
                input_dim=channels[1],
                hidden_dim=hidden_dim,
                prompt_dim=prompt_dim,
                use_prompt_cond=use_prompt_cond,
                use_neg_prompt_cond=use_neg_prompt_cond,
                use_block_query=use_block_query,
                num_heads=num_heads[1],
            )
        ),
        "up_blocks.3.attentions.0.transformer_blocks.0.attn2": AdapterConfig(
            adapter=CrossAttentionCFGAdapter,
            kwargs=dict(
                input_dim=channels[0],
                hidden_dim=hidden_dim,
                prompt_dim=prompt_dim,
                use_prompt_cond=use_prompt_cond,
                use_neg_prompt_cond=use_neg_prompt_cond,
                use_block_query=use_block_query,
                num_heads=num_heads[0],
            )
        ),
        "up_blocks.3.attentions.1.transformer_blocks.0.attn2": AdapterConfig(
            adapter=CrossAttentionCFGAdapter,
            kwargs=dict(
                input_dim=channels[0],
                hidden_dim=hidden_dim,
                prompt_dim=prompt_dim,
                use_prompt_cond=use_prompt_cond,
                use_neg_prompt_cond=use_neg_prompt_cond,
                use_block_query=use_block_query,
                num_heads=num_heads[0],
            )
        ),
        "up_blocks.3.attentions.2.transformer_blocks.0.attn2": AdapterConfig(
            adapter=CrossAttentionCFGAdapter,
            kwargs=dict(
                input_dim=channels[0],
                hidden_dim=hidden_dim,
                prompt_dim=prompt_dim,
                use_prompt_cond=use_prompt_cond,
                use_neg_prompt_cond=use_neg_prompt_cond,
                use_block_query=use_block_query,
                num_heads=num_heads[0],
            )
        ),
    }

    return inject_adapters(unet, adapters)
