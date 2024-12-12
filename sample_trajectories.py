import torch
import numpy as np
from diffusers import DiffusionPipeline, DiTPipeline
from tqdm import tqdm
import argparse
import random
import yaml
import os

from src.supported_models import SUPPORTED_MODELS, TEXT_MODELS


def main(args):
    is_text_model = args.base_model in TEXT_MODELS

    assert args.base_model in SUPPORTED_MODELS, f"base model not supported: {args.base_model}"
    assert not is_text_model or args.prompt_file is not None, "prompt file required for text models"

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if is_text_model:
        # Read prompts
        with open(args.prompt_file, "r") as f:
            prompts = yaml.safe_load(f)
            conditions = prompts["prompts"]

        # Ensure that each condition has a prompt
        for condition in conditions:
            assert "prompt" in condition, "condition must contain a 'prompt' key."
    else:
        conditions = [{ "class_label": label } for label in torch.arange(0, 1000).repeat(args.num_per_class)]

    # Load model
    pipe = DiffusionPipeline.from_pretrained(args.base_model).to(device)
    save_wrapper = SaveCFGTrajectoryUnetWrapper(pipe, args.inference_steps)

    if hasattr(pipe, "unet"):
        pipe.unet = save_wrapper
    elif hasattr(pipe, "transformer"):
        pipe.transformer = save_wrapper
    else:
        raise ValueError("pipeline does not have a model")
    
    # Create output directory
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

    # Save step -> timestep mapping
    pipe.scheduler.set_timesteps(args.inference_steps)
    torch.save(pipe.scheduler.timesteps, os.path.join(args.output_dir, "timesteps.pt"))

    # Save shape of latent space
    torch.save(
        [save_wrapper.model.config.in_channels, save_wrapper.model.config.sample_size, save_wrapper.model.config.sample_size],
        os.path.join(args.output_dir, "latent_shape.pt"),
    )

    n_sampled = 0
    for condition in tqdm(conditions):
        cfg_scale = random.uniform(args.min_cfg_scale, args.max_cfg_scale)
        save_wrapper.guidance_scale = cfg_scale

        sample_dir = os.path.join(args.output_dir, str(n_sampled).zfill(6))
        os.mkdir(sample_dir)

        if is_text_model:
            # Get conditioning variables (SDXL outputs 4 values with the first 2 being pos and neg
            # prompt, SD outputs 2 values; pos and neg prompt)
            embeds = pipe.encode_prompt(
                condition["prompt"],
                device,
                num_images_per_prompt=1,
                do_classifier_free_guidance=True,
                negative_prompt=condition.get("neg_prompt", None),
            )
            prompt_embeds, neg_prompt_embeds = embeds[0], embeds[1]

            pipe(
                prompt=condition["prompt"],
                neg_prompt=condition.get("neg_prompt", None),
                guidance_scale=cfg_scale,
                num_inference_steps=args.inference_steps,
            )

            # Save conditioning variables
            torch.save({
                "prompt": prompt_embeds[0].cpu(),
                "neg_prompt": neg_prompt_embeds[0].cpu(),
                "cfg_scale": cfg_scale,
                "additional_model_kwargs": save_wrapper.additional_kwargs,
            }, os.path.join(sample_dir, "conditioning.pt"))
        else:
            class_label = condition["class_label"]
            pipe(class_labels=[class_label], guidance_scale=cfg_scale, num_inference_steps=args.inference_steps)

            # Save conditioning variables
            torch.save({
                "class_label": class_label,
                "cfg_scale": cfg_scale,
                "additional_model_kwargs": save_wrapper.additional_kwargs,
            }, os.path.join(sample_dir, "conditioning.pt"))

        # Save and reset
        trajectory = save_wrapper.trajectories.numpy()
        fp = np.memmap(os.path.join(os.path.join(sample_dir, "trajectory.npy")), dtype=np.float32, mode="w+", shape=trajectory.shape)
        fp[:] = trajectory[:]
        fp.flush()

        save_wrapper.reset()
        n_sampled += 1


class SaveCFGTrajectoryUnetWrapper:
    def __init__(self,  pipe: DiffusionPipeline, num_steps: int):
        self.pipe = pipe

        if hasattr(pipe, "unet"):
            self.model = pipe.unet
            self.cond_dim = 1
        elif hasattr(pipe, "transformer"):
            self.model = pipe.transformer
            self.cond_dim = 0
        else:
            raise ValueError("pipeline does not have a model")

        self.channels = self.model.config.in_channels
        self.sample_size = self.model.config.sample_size

        self.guidance_scale = -1
        self.num_steps = num_steps
        self.reset()

    def __call__(self, *args, **kwargs):
        output = self.model(*args, **kwargs)

        # Save positional arguments, removing input and timestep
        if len(args) == 0:
            latent_input = kwargs["sample"]
        else:
            latent_input = args[0]

        # Save model input
        self.trajectories[self.step, 0] = remove_uncond_dim(latent_input, keep_dim=self.cond_dim).cpu()

        # Save model output
        if isinstance(output, tuple):
            noise_pred = output[0]
        else:
            noise_pred = output.sample

        # Remove learned sigma if present
        if noise_pred.shape[1] == 2 * self.channels:
            noise_pred, _ = noise_pred.chunk(2, dim=1)

        # Do classifier-free guidance (apparently DiT and stable diffusion have different ordering of
        # conditional and unconditional...)
        if isinstance(self.pipe, DiTPipeline):
            # https://github.com/huggingface/diffusers/blob/89e4d6219805975bd7d253a267e1951badc9f1c0/src/diffusers/pipelines/dit/pipeline_dit.py#L195-L203
            eps_cond, eps_uncond = noise_pred.chunk(2)
        else:
            # https://github.com/huggingface/diffusers/blob/89e4d6219805975bd7d253a267e1951badc9f1c0/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion.py#L1030-L1033
            eps_uncond, eps_cond = noise_pred.chunk(2)

        # Save CFG output
        eps_cfg = eps_uncond + self.guidance_scale * (eps_cond - eps_uncond)
        self.trajectories[self.step, 1] = eps_cfg.cpu()

        # Save additional arguments that need to be passed to the model during training
        if self.additional_kwargs is None:
            self.additional_kwargs = kwargs
            # Remove information that is already saved or unnecessary
            for key in ["sample", "timestep", "encoder_hidden_states", "class_labels", "return_dict"]:
                self.additional_kwargs.pop(key, None)

            self.additional_kwargs = remove_none(self.additional_kwargs)
            self.additional_kwargs = remove_uncond_dim(self.additional_kwargs, keep_dim=self.cond_dim)
            self.additional_kwargs = to_cpu(self.additional_kwargs)

        # Increase step
        self.step += 1

        return output

    def reset(self):
        self.additional_kwargs = None
        self.trajectories = torch.zeros((self.num_steps, 2, self.channels, self.sample_size, self.sample_size))
        self.step = 0

    def __getattr__(self, name):
        return getattr(self.model, name)


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


def remove_none(data):
    if isinstance(data, dict):
        return { k: remove_none(v) for k, v in data.items() if v is not None }

    if isinstance(data, list):
        return [remove_none(v) for v in data if v is not None]
    
    if isinstance(data, tuple):
        return tuple([remove_none(v) for v in data if v is not None])
    
    return data


def remove_uncond_dim(data, keep_dim=1):
    if isinstance(data, torch.Tensor):
        return data[keep_dim]
    
    if isinstance(data, dict):
        return { k: remove_uncond_dim(v, keep_dim) for k, v in data.items() }

    if isinstance(data, list):
        return [remove_uncond_dim(v, keep_dim) for v in data]
    
    if isinstance(data, tuple):
        return tuple([remove_uncond_dim(v, keep_dim) for v in data])

    raise ValueError(f"unsupported data type: {type(data)}")


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
