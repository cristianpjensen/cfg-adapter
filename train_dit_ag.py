import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from accelerate import Accelerator, load_checkpoint_and_dispatch
from accelerate.utils import set_seed
from accelerate.logging import get_logger
from glob import glob
from time import time
import warnings
import argparse
import logging
import os

from dit_ag import DiT_AG
from trajectory_dataset import TrajectoryDataset


torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def main(args):
    """Trains a new DiT CFG adapter model with synthetic data."""

    assert torch.cuda.is_available(), "training requires at least one gpu"
    set_seed(args.seed)

    experiment_index = len(glob(f"{args.results_dir}/*"))
    experiment_dir = os.path.join(args.results_dir, f"{experiment_index:03d}-DiT-AG")
    os.makedirs(experiment_dir, exist_ok=True)
    logger = create_logger(experiment_dir)

    # Set up accelerator
    with warnings.catch_warnings(action="ignore", category=FutureWarning):
        # Warning: `torch.cuda.amp.GradScalar(args...)` is deprecated
        accelerator = Accelerator(project_dir=experiment_dir)
        device = accelerator.device

    logger.info(f"experiment directory created at {experiment_dir}")

    assert args.image_size % 8 == 0, "image size must be divisible by 8"
    latent_size = args.image_size // 8

    # Create model and load pretrained weights
    model = DiT_AG(input_size=latent_size, num_classes=args.num_classes)

    # Load pretrained weights of base model (that are frozen)
    with warnings.catch_warnings(action="ignore", category=FutureWarning):
        # Warning: `weights_only=False` is not recommended
        ckpt_path = args.pretrained_ckpt or os.path.join("pretrained_models", f"DiT-XL-2-{args.image_size}x{args.image_size}.pt")
        model = load_checkpoint_and_dispatch(model, ckpt_path)

    # Optimizer
    opt = torch.optim.AdamW(model.adapters.parameters(), lr=1e-4, weight_decay=0)
    model.train()

    # Log number of parameters
    logger.info(f"adapter parameters: {sum(p.numel() for p in model.adapters.parameters()):,}")
    
    # Setup data
    dataset = TrajectoryDataset(args.data_path, num_timesteps=args.diffusion_timesteps)
    loader = DataLoader(
        dataset,
        batch_size=int(args.global_batch_size // accelerator.num_processes),
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )
    logger.info(f"dataset contains {len(dataset):,} points ({args.data_path})")

    # Prepare model, optimizer, and loader
    loader, model, opt = accelerator.prepare(loader, model, opt)

    # Variables for monitoring/logging purposes:
    train_steps = 0
    log_steps = 0
    running_loss = 0
    start_time = time()

    logger.info(f"training for {args.epochs} epochs...")

    for epoch in range(args.epochs):
        logger.info(f"beginning epoch {epoch}...")

        for teacher_input, teacher_output, timestep, metadata in loader:
            class_label = metadata["class_label"].int()
            cfg_scale = metadata["cfg_scale"].float()

            model_output = model.forward(teacher_input, timestep, class_label, cfg_scale)
            eps, _ = model_output.chunk(2, dim=1)
            loss = F.mse_loss(eps, teacher_output)

            # Backward pass
            opt.zero_grad()
            accelerator.backward(loss)
            opt.step()

            # Log loss values
            running_loss += loss.item()
            log_steps += 1
            train_steps += 1

            if train_steps % args.log_every == 0:
                # Measure training speed
                torch.cuda.synchronize()
                end_time = time()
                steps_per_sec = log_steps / (end_time - start_time)

                # Reduce loss history over all processes
                avg_loss = running_loss / (log_steps * accelerator.num_processes)

                # Log metrics
                logger.info(f"(step={train_steps:07d}) train Loss: {avg_loss:.4f}, train steps/sec: {steps_per_sec:.2f}")

                # Reset monitoring variables
                running_loss = 0
                log_steps = 0
                start_time = time()

            # Save checkpoint
            if train_steps % args.ckpt_every == 0 and train_steps > 0:
                accelerator.save_state(os.path.join(experiment_dir, "checkpoints", f"{train_steps:07d}"))

    logger.info("finished training")


def create_logger(logging_dir: os.PathLike) -> logging.Logger:
    """Create a logger that writes to a log file and stdout."""

    logging.basicConfig(
        level=logging.INFO,
        format="[\033[34m%(asctime)s\033[0m] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(), logging.FileHandler(os.path.join(logging_dir, "log.txt"))]
    )
    return get_logger(__name__)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--diffusion-timesteps", type=int, default=1000)
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--pretrained-ckpt", type=str, default=None)
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--epochs", type=int, default=1_400)
    parser.add_argument("--global-batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--ckpt-every", type=int, default=50_000)
    args = parser.parse_args()
    main(args)
