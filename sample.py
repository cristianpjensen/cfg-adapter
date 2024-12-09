import torch
from torchvision.utils import save_image
from diffusers import DiffusionPipeline
from safetensors import safe_open
import argparse
import yaml
import os

from src.models import get_adapter_unet
from src.supported_models import SUPPORTED_MODELS, TEXT_MODELS


def main(args):
    torch.manual_seed(args.seed)
    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    with open(os.path.join(args.results_dir, "config.yaml"), "r") as f:
        train_args = yaml.safe_load(f)

    is_text_model = train_args["base_model"] in TEXT_MODELS

    assert train_args["base_model"] in SUPPORTED_MODELS, f"base model not supported: {train_args['base_model']}"
    assert not is_text_model or args.prompt is not None, "negative prompt required for text models"
    assert is_text_model or args.class_label is not None, "class label required for imagenet models"

    pipe = DiffusionPipeline.from_pretrained(train_args["base_model"]).to(device)

    if is_text_model:
        # Get conditioning variables (SDXL outputs 4 values with the first 2 being pos and neg
        # prompt, SD outputs 2 values; pos and neg prompt)
        embeds = pipe.encode_prompt(
            args.prompt,
            device,
            negative_prompt=args.neg_prompt,
            num_images_per_prompt=args.num_images,
            do_classifier_free_guidance=True,
        )
        prompt_embeds, neg_prompt_embeds = embeds[0], embeds[1]
    else:
        class_label = args.class_label

    if args.use_adapter:
        unet = get_adapter_unet(
            model_name=train_args["base_model"],
            hidden_dim=train_args["hidden_dim"],
            use_prompt_cond=train_args["use_prompt_cond"],
            use_neg_prompt_cond=train_args["use_neg_prompt_cond"],
            use_block_query=train_args["use_block_query"],
        ).to(device)

        with safe_open(os.path.join(args.results_dir, "checkpoints", args.checkpoint, "model.safetensors"), framework="pt") as f:
            for name, param in unet.named_parameters():
                param.copy_(f.get_tensor(name).to(device))

        pipe.unet = unet
        pipe.unet.set_adapter_kwargs(
            cfg_scale=torch.tensor([args.cfg_scale] * args.num_images, device=device),
            prompt=prompt_embeds,
            neg_prompt=neg_prompt_embeds,
        )

    samples = pipe(
        prompt_embeds=prompt_embeds,
        neg_prompt_embeds=neg_prompt_embeds,
        num_inference_steps=args.inference_steps,
        guidance_scale=1.0 if args.use_adapter or not args.use_cfg else args.cfg_scale,
        output_type="pt",
    ).images.cpu()

    # Save images
    save_image(samples, args.output_file, nrow=args.num_rows, normalize=True, value_range=(0, 1))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output-file", type=str, required=True)
    parser.add_argument("--cfg-scale", type=float, required=True)
    parser.add_argument("--prompt", type=str, default=None)
    parser.add_argument("--neg-prompt", type=str, default=None)
    parser.add_argument("--class-label", type=int, choices=list(range(1000)), default=None)
    parser.add_argument("--no-adapter", action="store_false", dest="use_adapter", help="Disable adapter and use CFG instead.")
    parser.add_argument("--no-cfg", action="store_false", dest="use_cfg", help="Disable CFG (only had influence if adapter is disabled).")
    parser.add_argument("--num-images", type=int, default=1)
    parser.add_argument("--num-rows", type=int, default=4)
    parser.add_argument("--inference-steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    main(args)
