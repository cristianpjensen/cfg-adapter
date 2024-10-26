# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Samples a large number of images from a pre-trained DiT model using DDP.
Subsequently saves a .npz file that can be used to compute FID and other
evaluation metrics via the ADM repo: https://github.com/openai/guided-diffusion/tree/main/evaluations

For a simple single-GPU/CPU sampling script, see sample.py.
"""

import torch
import torch.distributed as dist
from diffusers.models import AutoencoderKL
from tqdm import tqdm
import os
from PIL import Image
import numpy as np
import math
import argparse
from glob import glob

from DiT.diffusion import create_diffusion
from dit_ag import DiT_AG


def main(args):
    """
    Run sampling.
    """

    assert torch.cuda.is_available(), "Sampling with DDP requires at least one GPU. sample.py supports CPU-only usage"
    torch.set_grad_enabled(False)

    # Setup DDP
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    seed = args.global_seed * dist.get_world_size() + rank

    torch.manual_seed(seed)
    torch.cuda.set_device(device)

    print(f"Starting rank={rank}, seed={seed}, world_size={dist.get_world_size()}.")

    # Load model
    latent_size = args.image_size // 8
    model = DiT_AG(depth=28, hidden_size=1152, patch_size=2, num_heads=16, input_size=latent_size, num_classes=args.num_classes).to(device)
    state_dict = torch.load(args.ckpt, weights_only=False)
    state_dict = state_dict["model"]
    model.load_state_dict(state_dict)
    model.eval()

    diffusion = create_diffusion(str(args.num_sampling_steps))
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)

    assert args.cfg_scale >= 1.0, "In almost all cases, cfg_scale be >= 1.0"

    # Create folder to save samples
    experiment_index = len(glob(f"{args.sample_dir}/*"))
    folder_name = f"{experiment_index:03d}-DiT_CFG_Adapter-size-{args.image_size}-vae-{args.vae}-cfg-{args.cfg_scale:.1f}-seed-{args.global_seed}"
    sample_folder_dir = f"{args.sample_dir}/{folder_name}"
    if rank == 0:
        os.makedirs(sample_folder_dir, exist_ok=True)
        print(f"Saving .png samples at {sample_folder_dir}")

    dist.barrier()

    # Figure out how many samples we need to generate on each GPU and how many iterations we need to run
    n = args.per_proc_batch_size
    global_batch_size = n * dist.get_world_size()

    # To make things evenly-divisible, we'll sample a bit more than we need and then discard the extra samples
    total_samples = int(math.ceil(args.num_fid_samples / global_batch_size) * global_batch_size)
    samples_needed_this_gpu = int(total_samples // dist.get_world_size())

    if rank == 0:
        print(f"Total number of images that will be sampled: {total_samples}")

    assert total_samples % dist.get_world_size() == 0, "total_samples must be divisible by world_size"
    assert samples_needed_this_gpu % n == 0, "samples_needed_this_gpu must be divisible by the per-GPU batch size"

    iterations = int(samples_needed_this_gpu // n)
    pbar = range(iterations)
    pbar = tqdm(pbar) if rank == 0 else pbar
    total = 0

    cfg_scale = torch.tensor([args.cfg_scale] * n, device=device)

    for _ in pbar:
        # Sample inputs:
        z = torch.randn(n, 4, latent_size, latent_size, device=device)
        y = torch.randint(0, args.num_classes, (n,), device=device)

        # Sample and convert to numpy array
        model_kwargs = dict(y=y, cfg_scale=cfg_scale)
        samples = diffusion.p_sample_loop(model.forward, z.shape, z, clip_denoised=False, model_kwargs=model_kwargs, progress=True, device=device)
        samples = vae.decode(samples / 0.18215).sample
        samples = torch.clamp(127.5 * samples + 128.0, 0, 255).permute(0, 2, 3, 1).to("cpu", dtype=torch.uint8).numpy()

        # Save samples to disk as individual .png files
        for i, sample in enumerate(samples):
            index = i * dist.get_world_size() + rank + total
            Image.fromarray(sample).save(f"{sample_folder_dir}/{index:06d}.png")

        total += global_batch_size

    # Make sure all processes have finished saving their samples before attempting to convert to .npz
    dist.barrier()
    if rank == 0:
        create_npz_from_sample_folder(sample_folder_dir, args.num_fid_samples)
        print("Done.")

    dist.barrier()
    dist.destroy_process_group()


def create_npz_from_sample_folder(sample_dir, num=50_000):
    """
    Builds a single .npz file from a folder of .png samples.
    """

    samples = []
    for i in tqdm(range(num), desc="Building .npz file from samples"):
        sample_pil = Image.open(f"{sample_dir}/{i:06d}.png")
        sample_np = np.asarray(sample_pil).astype(np.uint8)
        samples.append(sample_np)
    samples = np.stack(samples)
    assert samples.shape == (num, samples.shape[1], samples.shape[2], 3)
    npz_path = f"{sample_dir}.npz"
    np.savez(npz_path, arr_0=samples)
    print(f"Saved .npz file to {npz_path} [shape={samples.shape}].")
    return npz_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, help="Path to checkpoint.")
    parser.add_argument("--vae",  type=str, choices=["ema", "mse"], default="ema")
    parser.add_argument("--sample-dir", type=str, default="samples")
    parser.add_argument("--per-proc-batch-size", type=int, default=32)
    parser.add_argument("--num-fid-samples", type=int, default=10_000)
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--cfg-scale",  type=float, default=4.0)
    parser.add_argument("--num-sampling-steps", type=int, default=250)
    parser.add_argument("--global-seed", type=int, default=0)

    args = parser.parse_args()
    main(args)
