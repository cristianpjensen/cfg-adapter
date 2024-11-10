import os
import csv
import torch
import math
import random
import argparse
from tqdm import tqdm
from diffusers import DiffusionPipeline


def main(args):
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

    timesteps = args.num_timesteps
    latent_channels = pipe.unet.config.in_channels
    latent_size = pipe.unet.config.sample_size

    # A bit of a hack so that we can access the input and CFG output of the model. It relies on the
    # variable names of the internal implementation---however, I do not see why these would change.
    pipe._callback_tensor_inputs.extend(["noise_pred", "latent_model_input"])

    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

    info_csv = csv.writer(open(os.path.join(args.output_dir, "info.csv"), "w+"))
    info_csv.writerow(["trajectory_filename", "cfg_scale", "prompt"])

    n_sampled = 0

    for prompt_batch in tqdm(batched(prompts, args.batch_size), total=math.ceil(len(prompts) / args.batch_size)):
        bs = len(prompt_batch)

        trajectories = torch.zeros((bs, timesteps, 2, latent_channels, latent_size, latent_size))
        cfg_scale = random.uniform(args.min_cfg_scale, args.max_cfg_scale)

        def save_trajectory_callback(pipe: DiffusionPipeline, step: int, timestep: torch.Tensor, callback_kwargs: dict):
            model_input, _ = callback_kwargs["latent_model_input"].chunk(2)
            model_cfg_output = callback_kwargs["noise_pred"]
            trajectories[:, timesteps - step - 1, 0] = model_input.cpu()
            trajectories[:, timesteps - step - 1, 1] = model_cfg_output.cpu()
            return callback_kwargs

        pipe(
            prompt_batch,
            guidance_scale=cfg_scale,
            num_inference_steps=timesteps,
            callback_on_step_end=save_trajectory_callback,
            callback_on_step_end_tensor_inputs=["noise_pred", "latent_model_input"],
        )

        # Save trajectories
        for i in range(bs):
            filename = f"{str(n_sampled).zfill(6)}.pt"
            torch.save(trajectories[i], os.path.join(args.output_dir, filename))
            info_csv.writerow([filename, cfg_scale, prompt_batch[i]])
            n_sampled += 1


def batched(datas, batch_size):
    for i in range(0, len(datas), batch_size):
        yield datas[i:i+batch_size]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="stabilityai/stable-diffusion-2-1")
    parser.add_argument("--num-timesteps", type=int, default=999)
    parser.add_argument("--prompt-path", type=str, required=True)
    parser.add_argument("--samples-per-prompt", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--min-cfg-scale", type=float, default=1.0)
    parser.add_argument("--max-cfg-scale", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", type=str, default="sd_trajectory_data")
    args = parser.parse_args()
    main(args)
