import torch
from diffusers import DiffusionPipeline
from tqdm import tqdm
import argparse
import random
import yaml
import os


def main(args):
    assert args.base_model in [
        "stabilityai/stable-diffusion-2-1",
        "facebook/DiT-XL-2-256",
        "facebook/DiT-XL-2-512",
    ], "base model not supported"
    text_model = args.base_model in ["stabilityai/stable-diffusion-2-1"]

    assert not text_model or args.prompt_file is not None, "prompt file required for text models"

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Read prompts
    if text_model:
        with open(args.prompt_file, "r") as f:
            prompts = yaml.safe_load(f)
            conditions = prompts["prompts"]
    else:
        conditions = [{ "class_label": label } for label in torch.arange(0, 1000).repeat(args.num_per_class)]

    # Load model
    pipe = DiffusionPipeline.from_pretrained(args.base_model).to(device)
    save_wrapper = SaveCFGTrajectoryUnetWrapper(pipe, args.inference_steps)
    pipe.unet = save_wrapper
    pipe.scheduler.set_timesteps(args.inference_steps)

    # Create output directory
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

    # Save step -> timestep mapping
    torch.save(pipe.scheduler.timesteps, os.path.join(args.output_dir, "timesteps.pt"))

    n_sampled = 0
    for condition in tqdm(conditions):
        cfg_scale = random.uniform(args.min_cfg_scale, args.max_cfg_scale)

        sample_dir = os.path.join(args.output_dir, str(n_sampled).zfill(6))
        os.mkdir(sample_dir)

        if text_model:
            # TODO: Add support for SDXL with `additional_model_kwargs` and `encode_prompt` being handled
            # properly
            assert "prompt" in condition, "condition must contain a 'prompt' key."

            # Get conditioning variables
            prompt_embeds, neg_prompt_embeds = pipe.encode_prompt(
                condition["prompt"],
                device,
                num_images_per_prompt=1,
                do_classifier_free_guidance=True,
                negative_prompt=condition.get("neg_prompt", None),
            )

            # Run model
            pipe(
                prompt_embeds=prompt_embeds,
                negative_prompt_embeds=neg_prompt_embeds,
                guidance_scale=cfg_scale,
                num_inference_steps=args.inference_steps,
            )

            # Save conditioning variables
            torch.save({
                "prompt": prompt_embeds[0],
                "neg_prompt": neg_prompt_embeds[0],
                "cfg_scale": cfg_scale,
            }, os.path.join(sample_dir, "conditioning.pt"))
        else:
            class_label = condition["class_label"]

            # Run model
            pipe(
                class_labels=[class_label],
                guidance_scale=cfg_scale,
                num_inference_steps=args.inference_steps,
            )

            # Save conditioning variables
            torch.save({
                "class_label": class_label,
                "cfg_scale": cfg_scale,
            }, os.path.join(sample_dir, "conditioning.pt"))

        # Save trajectory
        torch.save(save_wrapper.trajectories, os.path.join(sample_dir, "trajectory.pt"))

        # Reset trajectory
        save_wrapper.reset()
        n_sampled += 1


class SaveCFGTrajectoryUnetWrapper:
    def __init__(self,  pipe: DiffusionPipeline, num_steps: int):
        self.pipe = pipe
        self.unet = pipe.unet
        self.num_steps = num_steps
        self.reset()

    def __call__(self, *args, **kwargs):
        output = self.unet(*args, **kwargs)

        # Save positional arguments, removing input and timestep
        if len(args) == 0:
            latent_input = kwargs["sample"]
        else:
            latent_input = args[0]

        # Save model input
        self.trajectories[self.step, 0] = remove_uncond_dim(latent_input).cpu()

        # Save model output
        if kwargs["return_dict"]:
            noise_pred = output.sample
        else:
            noise_pred = output[0]

        # Do classifier-free guidance
        # https://github.com/huggingface/diffusers/blob/89e4d6219805975bd7d253a267e1951badc9f1c0/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion.py#L1030C17-L1037C120
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        cfg_pred = noise_pred_uncond + self.pipe.guidance_scale * (noise_pred_text - noise_pred_uncond)

        self.trajectories[self.step, 1] = cfg_pred.cpu()

        # Increase step
        self.step += 1

        return output

    def reset(self):
        self.trajectories = torch.zeros((self.num_steps, 2, self.unet.config.in_channels, self.unet.config.sample_size, self.unet.config.sample_size))
        self.step = 0

    def __getattr__(self, name):
        return getattr(self.unet, name)


def to_cpu(data):
    if hasattr(data, "to"):
        return data.cpu()
    
    if isinstance(data, dict):
        return { k: to_cpu(v) for k, v in data.items() }

    if isinstance(data, list):
        return [to_cpu(v) for v in data]
    
    if isinstance(data, tuple):
        return tuple([to_cpu(v) for v in data])
    
    return data 


def remove_uncond_dim(data):
    if isinstance(data, torch.Tensor):
        return data[1]
    
    if isinstance(data, dict):
        return { k: remove_uncond_dim(v) for k, v in data.items() }

    if isinstance(data, list):
        return [remove_uncond_dim(v) for v in data]
    
    if isinstance(data, tuple):
        return tuple([remove_uncond_dim(v) for v in data])

    raise ValueError(f"Unsupported data type: {type(data)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--base-model", type=str, default="stabilityai/stable-diffusion-2-1")
    parser.add_argument("--prompt-file", type=str, default=None, help="YAML file containing prompts (required for text models).")
    parser.add_argument("--num-per-class", type=int, default=1, help="Number of samples per class (only used for imagenet models).")
    parser.add_argument("--inference-steps", type=int, default=999)
    parser.add_argument("--min-cfg-scale", type=float, default=1.0)
    parser.add_argument("--max-cfg-scale", type=float, default=15.0)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    main(args)
