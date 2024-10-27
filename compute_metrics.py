"""Compute FID from a checkpoint."""


import os
import torch
from torchvision.io import write_png, read_image
import numpy as np
import logging
import argparse
from glob import glob
from tqdm import tqdm
from safetensors import safe_open
from diffusers.models import AutoencoderKL

from DiT.diffusion import create_diffusion
from dit_ag import DiT_AG
from evaluator import evaluate


def main(args):
    """Samples from a trained DiT AG model and computes the FID score."""

    assert torch.cuda.is_available(), "sampling requires at least one gpu"
    torch.set_grad_enabled(False)
    torch.manual_seed(args.seed)
    device = "cuda"

    # Create experiment directory
    experiment_index = len(glob(f"{args.output_dir}/*"))
    experiment_dir = os.path.join(args.output_dir, f"{experiment_index:03d}-DiT-AG-cfg_{args.cfg_scale:.2f}-seed_{args.seed}")
    samples_dir = os.path.join(experiment_dir, "samples")
    os.makedirs(samples_dir, exist_ok=True)
    logger = create_logger(experiment_dir)

    logger.info(f"experiment directory created at {experiment_dir}")
    logger.info(f"saving .png samples at {samples_dir}")

    # Load model
    logger.info("loading model...")
    latent_size = args.image_size // 8
    model = DiT_AG(input_size=latent_size, num_classes=args.num_classes)

    with safe_open(os.path.join(args.ckpt, "model.safetensors"), framework="pt") as f:
        for name, param in model.named_parameters():
            if name in f.keys():
                param.copy_(f.get_tensor(name))
            else:
                logger.warning(f"parameter {name} not found in checkpoint")

    model.to(device)
    model.eval()

    logger.info("loading vae...")
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)
    diffusion = create_diffusion(str(args.num_sampling_steps))

    # Sample and save images
    for i in tqdm(range(0, args.num_samples, args.batch_size), desc="sampling"):
        bs = min(args.batch_size, args.num_samples - i)
        z = torch.randn(bs, 4, latent_size, latent_size, device=device)
        y = torch.randint(0, args.num_classes, (bs,), device=device)
        cfg_scale = torch.tensor([args.cfg_scale] * bs, device=device)

        # Sample
        model_kwargs = dict(y=y, cfg_scale=cfg_scale)
        samples = diffusion.p_sample_loop(model.forward, z.shape, z, clip_denoised=False, model_kwargs=model_kwargs, progress=True, device=device)
        samples = vae.decode(samples / 0.18215).sample
        samples = torch.clamp(127.5 * samples + 128.0, 0, 255).cpu().byte()

        for j, sample in enumerate(samples):
            index = i + j
            write_png(sample, os.path.join(samples_dir, f"{index:06d}.png"))

    # Convert to npz for evaluation
    samples = []
    for i in tqdm(range(args.num_samples), desc="converting to npz for evaluation"):
        sample = read_image(os.path.join(samples_dir, f"{i:06d}.png")).permute(1, 2, 0).numpy()
        samples.append(sample)

    samples = np.stack(samples, axis=0)
    assert samples.shape == (args.num_samples, args.image_size, args.image_size, 3)

    npz_path = os.path.join(experiment_dir, "samples.npz")
    np.savez(npz_path, arr_0=samples)

    logger.info(f"samples saved to {npz_path} [shape={samples.shape}]")

    logger.info("evaluating metrics...")
    metrics = evaluate(args.ref_batch, npz_path)

    logger.info(f" - inception score: {metrics['inception_score']:.5f}")
    logger.info(f" - FID: {metrics['fid']:.5f}")
    logger.info(f" - sFID: {metrics['sfid']:.5f}")
    logger.info(f" - precision: {metrics['precision']:.5f}")
    logger.info(f" - recall: {metrics['recall']:.5f}")

    logger.info("done")


def create_logger(logging_dir: os.PathLike) -> logging.Logger:
    """Create a logger that writes to a log file and stdout."""

    logging.basicConfig(
        level=logging.INFO,
        format="[\033[34m%(asctime)s\033[0m] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(), logging.FileHandler(os.path.join(logging_dir, "log.txt"))]
    )
    return logging.getLogger(__name__)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True, help="Path to checkpoint.")
    parser.add_argument("--output-dir", type=str, required=True, help="Directory to save samples.")
    parser.add_argument("--ref-batch", type=str, required=True, help="path to reference batch npz file")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="ema")
    parser.add_argument("--num-samples", type=int, default=50_000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--cfg-scale", type=float, default=1.5)
    parser.add_argument("--num-sampling-steps", type=int, default=250)
    args = parser.parse_args()
    main(args)
