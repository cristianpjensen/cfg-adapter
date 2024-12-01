import torch
from diffusers import DiffusionPipeline
from tqdm import tqdm
import argparse
import random
import yaml
import os


torch.set_float32_matmul_precision("high")


def main(args):
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Read prompts
    with open(args.prompt_file, "r") as f:
        prompts = yaml.safe_load(f)
        prompts = prompts["prompts"]

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
    for prompt in tqdm(prompts):
        assert "prompt" in prompt, "prompt must contain a 'prompt' key."

        # Forward pass
        cfg_scale = random.uniform(args.min_cfg_scale, args.max_cfg_scale)
        pipe(prompt["prompt"], negative_prompt=prompt.get("neg_prompt", None) if args.use_neg_prompt else None, guidance_scale=cfg_scale, num_inference_steps=args.inference_steps)

        # Save trajectory and other information necessary for the forward pass
        sample_dir = os.path.join(args.output_dir, str(n_sampled).zfill(6))
        os.mkdir(sample_dir)
        torch.save(save_wrapper.trajectories, os.path.join(sample_dir, "trajectory.pt"))
        torch.save({ "cfg_scale": cfg_scale }, os.path.join(sample_dir, "adapter_kwargs.pt"))
        torch.save(save_wrapper.get_arguments(), os.path.join(sample_dir, "model_arguments.pt"))
        n_sampled += 1

        # Reset args, kwargs, and trajectories
        save_wrapper.reset()


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
            self.args = []
        elif len(args) == 1:
            latent_input = args[0]
            self.args = []
        else:
            latent_input = args[0]
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

        # Decrease step
        self.step -= 1

        return output

    def get_arguments(self):
        return {
            "args": self.args,
            "kwargs": self.kwargs,
        }

    def reset(self):
        self.trajectories = torch.zeros((self.num_steps, 2, self.unet.config.in_channels, self.unet.config.sample_size, self.unet.config.sample_size))
        self.args = None
        self.kwargs = None
        self.step = self.num_steps - 1

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
    parser.add_argument("--prompt-file", type=str, required=True)
    parser.add_argument("--base-model", type=str, default="stabilityai/stable-diffusion-2-1")
    parser.add_argument("--inference-steps", type=int, default=200)
    parser.add_argument("--min-cfg-scale", type=float, default=1.0)
    parser.add_argument("--max-cfg-scale", type=float, default=15.0)
    parser.add_argument("--no-neg-prompt", action="store_false", dest="use_neg_prompt", help="Disable negative prompting.")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    main(args)
