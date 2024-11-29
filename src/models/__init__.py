from .sd import get_sd_ag_unet


def get_adapter_unet(model_name: str):
    match model_name:
        case "stabilityai/stable-diffusion-2-1":
            return get_sd_ag_unet
        case _:
            raise ValueError(f"unsupported model: {model_name}")
