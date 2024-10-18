# Adapted from the sampling script of DiT repository

"""
Sample full trajectories from a pre-trained DiT, using CFG. These form a dataset to be used for
training a student CFG adapter model.
"""

import os
import csv
import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import argparse
from tqdm import tqdm

from DiT.diffusion import create_diffusion
from DiT.download import find_model
from DiT.models import DiT_models


def main(args):
    torch.manual_seed(args.seed)
    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.ckpt is None:
        assert args.model == "DiT-XL/2", "Only DiT-XL/2 models are available for auto-download."
        assert args.image_size in [256, 512]
        assert args.num_classes == 1000

    # Load model
    latent_size = args.image_size // 8
    model = DiT_models[args.model](
        input_size=latent_size,
        num_classes=args.num_classes
    ).to(device)

    # Auto-download a pre-trained model or load a custom DiT checkpoint from DiT/train.py
    ckpt_path = args.ckpt or f"DiT-XL-2-{args.image_size}x{args.image_size}.pt"
    state_dict = find_model(ckpt_path)
    model.load_state_dict(state_dict)
    model.eval()
    diffusion = create_diffusion("")

    n_sampled = 0
    info_csv = csv.writer(open(os.path.join(args.output_dir, "info.csv"), "w"))
    for cfg_scale in args.cfg_scales:
        for class_label in tqdm(range(args.num_classes), desc=f"CFG scale: {cfg_scale:.3f}", leave=False):
            # Sample `args.samples_per_class` trajectories for each class, `args.batch_size` at a time
            while n_sampled < args.samples_per_class:
                bs = min(args.batch_size, args.samples_per_class - n_sampled)
                z = torch.randn(bs, 4, latent_size, latent_size, device=device)
                y = torch.tensor([class_label] * bs, device=device)
                y_null = torch.tensor([1000] * bs, device=device)

                z = torch.cat([z, z], dim=0)
                y = torch.cat([y, y_null], dim=0)
                model_kwargs = dict(y=y, cfg_scale=cfg_scale)

                trajectories = torch.zeros((bs, diffusion.num_timesteps, 4, latent_size, latent_size))

                for i, sample in tqdm(
                    enumerate(
                        diffusion.p_sample_loop_progressive(
                            model.forward_with_cfg, z.shape, z, clip_denoised=False,
                            model_kwargs=model_kwargs, progress=False, device=device,
                        )
                    ),
                    leave=False,
                    total=diffusion.num_timesteps,
                ):
                    # The first half of the samples are the CFG trajectory noises
                    sample, _ = sample["sample"].chunk(2, dim=0)
                    trajectories[:, i] = sample.cpu()

                for i in range(bs):
                    filename = f"{str(n_sampled).zfill(6)}.pt"
                    torch.save(trajectories[i], os.path.join(args.output_dir, filename))
                    info_csv.writerow([filename, class_label, cfg_scale])
                    n_sampled += 1

    print(f"Saved {n_sampled} trajectories to {args.output_dir}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, choices=list(DiT_models.keys()), default="DiT-XL/2")
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--samples-per-class", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--cfg-scales", nargs="+", type=float, default=[1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ckpt", type=str, default=None, help="Optional path to a teacher DiT checkpoint (default: auto-download a pre-trained DiT-XL/2 model).")
    parser.add_argument("--output-dir", type=str, default="trajectory_data")
    args = parser.parse_args()
    main(args)
