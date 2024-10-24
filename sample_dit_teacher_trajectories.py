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

    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

    info_csv = csv.writer(open(os.path.join(args.output_dir, "info.csv"), "w+"))
    info_csv.writerow(["trajectory_filename", "cfg_scale", "class_label"])

    n_sampled = 0
    class_labels = torch.arange(0, args.num_classes, step=1 / args.samples_per_class).int()

    for class_label in tqdm(class_labels.split(args.batch_size)):
        bs = class_label.shape[0]

        # Sample latents
        z = torch.randn(bs, 4, latent_size, latent_size, device=device)
        y = class_label.to(device)
        y_null = torch.tensor([1000] * bs, device=device)

        # Sample CFG scales from uniform distribution over [min_cfg_scale, max_cfg_scale]
        cfg_scale = torch.rand((bs, 1, 1, 1), device=device) * (args.max_cfg_scale - args.min_cfg_scale) + args.min_cfg_scale
        z = torch.cat([z, z], dim=0)
        y = torch.cat([y, y_null], dim=0)

        # Define trajectory loop
        model_kwargs = dict(y=y, cfg_scale=cfg_scale)
        diffusion_loop = diffusion.p_sample_loop_progressive(
            model.forward_with_cfg, z.shape, z, clip_denoised=False,
            model_kwargs=model_kwargs, progress=False, device=device
        )

        # Keep track of trajectories
        prev_sample, _ = z.chunk(2, dim=0)
        trajectory = torch.zeros(bs, diffusion.num_timesteps, 2, 4, latent_size, latent_size)

        for t, sample in tqdm(enumerate(diffusion_loop), total=diffusion.num_timesteps, leave=False):
            # Compute epsilon prediction for the current timestep
            timestep = diffusion.num_timesteps - t - 1
            timestep_tensor = torch.tensor([timestep] * bs, dtype=torch.long, device=device)
            pred_xstart, _ = sample["pred_xstart"].chunk(2, dim=0)
            eps = diffusion._predict_eps_from_xstart(prev_sample, timestep_tensor, pred_xstart)

            # Save model in- and output of the current timestep into the trajectory
            trajectory[:, timestep] = torch.stack([prev_sample, eps]).transpose(0, 1).cpu()

            # Update previous sample
            prev_sample, _ = sample["sample"].chunk(2, dim=0)

        for j in range(bs):
            filename = f"{str(n_sampled).zfill(6)}.pt"
            torch.save(trajectory[j], os.path.join(args.output_dir, filename))
            info_csv.writerow([filename, cfg_scale[j].item(), class_label])
            n_sampled += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, choices=list(DiT_models.keys()), default="DiT-XL/2")
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--samples-per-class", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--min-cfg-scale", type=float, default=1.0)
    parser.add_argument("--max-cfg-scale", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ckpt", type=str, default=None, help="Optional path to a teacher DiT checkpoint (default: auto-download a pre-trained DiT-XL/2 model).")
    parser.add_argument("--output-dir", type=str, default="trajectory_data")
    args = parser.parse_args()
    main(args)
