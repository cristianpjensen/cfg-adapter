from .sd import get_sd_adapter_unet
from .dit import get_dit_adapter_transformer


def get_adapter_unet(
    model_name: str,
    hidden_dim: int=320,
    use_prompt_cond=False,
    use_neg_prompt_cond=False,
):
    match model_name:
        case "stabilityai/stable-diffusion-2-1":
            return get_sd_adapter_unet(
                model_name,
                hidden_dim=hidden_dim,
                use_prompt_cond=use_prompt_cond,
                use_neg_prompt_cond=use_neg_prompt_cond,
            )
        case "stabilityai/stable-diffusion-xl-base-1.0":
            return get_sd_adapter_unet(
                model_name,
                hidden_dim=hidden_dim,
                use_prompt_cond=use_prompt_cond,
                use_neg_prompt_cond=use_neg_prompt_cond,
            )
        case "facebook/DiT-XL-2-256":
            return get_dit_adapter_transformer(hidden_dim=hidden_dim, image_size=256)
        case "facebook/DiT-XL-2-512":
            return get_dit_adapter_transformer(hidden_dim=hidden_dim, image_size=512)
        case _:
            raise ValueError(f"unsupported model: {model_name}")
