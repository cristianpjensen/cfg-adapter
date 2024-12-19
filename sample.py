import torch
from torchvision.utils import save_image
from diffusers import DiffusionPipeline, DiTPipeline
from glob import glob
import argparse
import yaml
import os

from src.models import get_adapter_unet
from src.supported_models import SUPPORTED_MODELS, TEXT_MODELS


def main(args):
    torch.manual_seed(args.seed)
    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load in config
    with open(os.path.join(args.result_dir, "config.yaml"), "r") as f:
        train_args = yaml.safe_load(f)

    # Take final checkpoint if none is specified
    if args.ckpt is None:
        checkpoints = glob(os.path.join(args.result_dir, "checkpoints", "*.pt"))
        if len(checkpoints) == 0:
            raise ValueError("No checkpoints found")

        args.ckpt = os.path.basename(max(checkpoints))
        print("using checkpoint:", args.ckpt)

    is_text_model = train_args["base_model"] in TEXT_MODELS
    num_images = args.num_images if is_text_model else len(args.class_labels)

    assert train_args["base_model"] in SUPPORTED_MODELS, f"base model not supported: {train_args['base_model']}"
    assert not is_text_model or args.prompt is not None, "negative prompt required for text models"
    assert is_text_model or args.class_labels is not None, "class label required for imagenet models"

    pipe = DiffusionPipeline.from_pretrained(train_args["base_model"]).to(device)

    if is_text_model:
        # Get conditioning variables (SDXL outputs 4 values with the first 2 being pos and neg
        # prompt, SD outputs 2 values; pos and neg prompt)
        embeds = pipe.encode_prompt(
            args.prompt,
            device,
            negative_prompt=args.neg_prompt,
            num_images_per_prompt=num_images,
            do_classifier_free_guidance=True,
        )
        prompt_embeds, neg_prompt_embeds = embeds[0], embeds[1]
        class_labels = None
    else:
        class_labels = args.class_labels
        prompt_embeds = None
        neg_prompt_embeds = None

    if args.use_adapter:
        model = get_adapter_unet(
            model_name=train_args["base_model"],
            hidden_dim=train_args["hidden_dim"],
            use_prompt_cond=train_args["use_prompt_cond"],
            use_neg_prompt_cond=train_args["use_neg_prompt_cond"],
        )

        sd = torch.load(
            os.path.join(args.result_dir, "checkpoints", args.ckpt),
            map_location=model.device,
            weights_only=False,
        )
        sd = sd["ema"]
        model.load_adapter_state_dicts(sd)

        model.set_adapter_kwargs(
            cfg_scale=torch.tensor([args.cfg_scale] * num_images, device=device),
            prompt=prompt_embeds,
            neg_prompt=neg_prompt_embeds,
            class_labels=class_labels,
        )

        if isinstance(pipe, DiTPipeline):
            pipe.transformer.cpu()
            pipe.transformer = model.to(device)
        else:
            pipe.unet.cpu()
            pipe.unet = model.to(device)

    pipe = pipe.to(device)

    if is_text_model:
        samples = pipe(
            prompt=args.prompt,
            negative_prompt=args.neg_prompt,
            num_inference_steps=args.inference_steps,
            num_images_per_prompt=num_images,
            guidance_scale=1.0 if args.use_adapter or not args.use_cfg else args.cfg_scale,
            output_type="pt",
        ).images.cpu()
    else:
        samples = pipe(
            class_labels=class_labels,
            num_inference_steps=args.inference_steps,
            guidance_scale=1.0 if args.use_adapter or not args.use_cfg else args.cfg_scale,
            output_type="np",
        ).images
        samples = torch.from_numpy(samples).permute(0, 3, 1, 2)

    # Save images
    save_image(samples, args.output_file, nrow=args.num_cols, normalize=True, value_range=(0, 1))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", type=str, required=True)
    parser.add_argument("--ckpt", type=int, required=None)
    parser.add_argument("--output-file", type=str, default="sample.png")
    parser.add_argument("--cfg-scale", type=float, default=4.0)
    parser.add_argument("--prompt", type=str, default=None)
    parser.add_argument("--neg-prompt", type=str, default=None)
    parser.add_argument("--class-labels", type=int, choices=list(range(1000)), nargs="+", default=None)
    parser.add_argument("--disable-adapter", action="store_false", dest="use_adapter", help="Disable adapter and use CFG instead.")
    parser.add_argument("--disable-cfg", action="store_false", dest="use_cfg", help="Disable CFG (only has influence if adapter is disabled).")
    parser.add_argument("--num-images", type=int, default=4)
    parser.add_argument("--num-cols", type=int, default=4)
    parser.add_argument("--inference-steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    main(args)
