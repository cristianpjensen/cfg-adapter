import os
import torch
import random
import argparse
from tqdm import tqdm
from diffusers import DiffusionPipeline


class SaveCFGTrajectoryUnetWrapper:
    def __init__(self,  pipe: DiffusionPipeline, timesteps: int):
        self.pipe = pipe
        self.unet = pipe.unet
        self.timesteps = timesteps
        self.reset()

    def __call__(self, *args, **kwargs):
        output = self.unet(*args, **kwargs)

        # Save positional arguments, removing input and timestep
        if len(args) == 0:
            latent_input = kwargs["sample"]
            timestep = kwargs["timestep"].long() - 1
            self.args = []
        elif len(args) == 1:
            latent_input = args[0]
            timestep = kwargs["timestep"].long() - 1
            self.args = []
        else:
            latent_input = args[0]
            timestep = args[1].long() - 1
            self.args = args[2:]

        # Save positional arguments
        if self.args is None:
            self.args = to_cpu(self.args)
            self.args = remove_uncond_dim(self.args)
        
        # Save keyword arguments
        if self.kwargs is None:
            self.kwargs = to_cpu({ k: v for k, v in kwargs.items() if k != "return_dict" and v is not None })
            self.kwargs = remove_uncond_dim(self.kwargs)

        # Save model input
        self.trajectories[timestep, 0] = remove_uncond_dim(latent_input).cpu()

        # Save model output
        if kwargs["return_dict"]:
            noise_pred = output.sample
        else:
            noise_pred = output[0]

        # Do classifier-free guidance
        # https://github.com/huggingface/diffusers/blob/89e4d6219805975bd7d253a267e1951badc9f1c0/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion.py#L1030C17-L1037C120
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        cfg_pred = noise_pred_uncond + self.pipe.guidance_scale * (noise_pred_text - noise_pred_uncond)

        self.trajectories[timestep, 1] = cfg_pred.cpu()

        return output

    def get_arguments(self):
        return {
            "args": self.args,
            "kwargs": self.kwargs,
        }

    def reset(self):
        self.trajectories = torch.zeros((self.timesteps, 2, self.unet.config.in_channels, self.unet.config.sample_size, self.unet.config.sample_size))
        self.args = None
        self.kwargs = None

    def __getattr__(self, name):
        return getattr(self.unet, name)


def main(args):
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Read prompts
    with open(args.prompt_path, "r") as f:
        prompts = f.readlines()

    # Duplicate every prompt to match the number of samples per prompt
    prompts = [prompt.strip() for prompt in prompts for _ in range(args.samples_per_prompt)]

    # Load model
    pipe = DiffusionPipeline.from_pretrained(args.model).to(device)
    save_wrapper = SaveCFGTrajectoryUnetWrapper(pipe, args.num_timesteps)
    pipe.unet = save_wrapper

    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

    n_sampled = 0

    for prompt in tqdm(prompts):
        cfg_scale = random.uniform(args.min_cfg_scale, args.max_cfg_scale)
        pipe(prompt, guidance_scale=cfg_scale, num_inference_steps=args.num_timesteps)

        # Save trajectory and other information necessary for the forward pass
        sample_dir = os.path.join(args.output_dir, str(n_sampled).zfill(6))
        os.mkdir(sample_dir)
        torch.save(save_wrapper.trajectories, os.path.join(sample_dir, "trajectory.pt"))
        torch.save({ "cfg_scale": cfg_scale }, os.path.join(sample_dir, "adapter_kwargs.pt"))
        torch.save(save_wrapper.get_arguments(), os.path.join(sample_dir, "model_arguments.pt"))
        n_sampled += 1

        # Reset args, kwargs, and trajectories
        save_wrapper.reset()


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
    parser.add_argument("--model", type=str, default="stabilityai/stable-diffusion-2-1")
    parser.add_argument("--num-timesteps", type=int, default=999)
    parser.add_argument("--prompt-path", type=str, required=True)
    parser.add_argument("--samples-per-prompt", type=int, default=1)
    parser.add_argument("--min-cfg-scale", type=float, default=1.0)
    parser.add_argument("--max-cfg-scale", type=float, default=15.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", type=str, default="sd_trajectory_data")
    args = parser.parse_args()
    main(args)
