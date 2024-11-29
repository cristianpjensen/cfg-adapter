import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from accelerate import Accelerator
from accelerate.utils import set_seed
from accelerate.logging import get_logger
from glob import glob
from time import time
import warnings
import argparse
import logging
import os

from sd_ag import GET_ADAPTER_UNET
from trajectory_dataset import TrajectoryDataset


def main(args):
    """Trains a new Stable Diffusion CFG adapter model with synthetic data."""

    assert torch.cuda.is_available(), "training requires at least one gpu"
    set_seed(args.seed)

    experiment_index = len(glob(f"{args.results_dir}/*"))
    experiment_dir = os.path.join(args.results_dir, f"{experiment_index:03d}")
    os.makedirs(experiment_dir, exist_ok=True)
    logger = create_logger(experiment_dir)

    # Set up accelerator
    with warnings.catch_warnings(action="ignore", category=FutureWarning):
        accelerator = Accelerator(project_dir=experiment_dir)

    logger.info(f"experiment directory created at {experiment_dir}")

    # Create model and load pretrained weights
    model_with_adapters = GET_ADAPTER_UNET[args.base_model]()
    opt = torch.optim.AdamW(model_with_adapters.adapters.parameters(), lr=1e-4, weight_decay=0)
    model_with_adapters.train_adapters()

    # Log number of parameters
    num_model_params = sum(p.numel() for p in model_with_adapters.parameters())
    num_adapter_params = sum(p.numel() for p in model_with_adapters.adapters.parameters())
    num_trainable_model_params = sum(p.numel() for p in model_with_adapters.model.parameters() if p.requires_grad)
    logger.info(f"total model parameters: {num_model_params:,}")
    logger.info(f"adapter parameters: {num_adapter_params:,}")
    logger.info(f"trainable model parameters: {num_trainable_model_params:,}")

    assert num_adapter_params == num_trainable_model_params, "adapter parameters should be the only trainable parameters"
    
    # Setup data
    dataset = TrajectoryDataset(args.data_path)
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
    loader, model_with_adapters, opt = accelerator.prepare(loader, model_with_adapters, opt)

    # Variables for monitoring/logging purposes:
    train_steps = 0
    log_steps = 0
    running_loss = 0
    start_time = time()

    logger.info(f"training for {args.epochs} epochs...")

    for epoch in range(args.epochs):
        logger.info(f"beginning epoch {epoch}...")

        for teacher_input, teacher_output, timestep, kwargs in loader:
            adapter_kwargs = kwargs["adapter_kwargs"]
            model_arguments = kwargs["model_arguments"]
            model_args = model_arguments["args"]
            model_kwargs = model_arguments["kwargs"]
            
            model_with_adapters.set_kwargs(cfg_scale=adapter_kwargs["cfg_scale"], encoder_hidden_states=model_kwargs["encoder_hidden_states"])
            model_output = model_with_adapters.forward(teacher_input, timestep, *model_args, **model_kwargs).sample
            loss = F.mse_loss(model_output, teacher_output)

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
                logger.info(f"(step={train_steps:07d}) train loss: {avg_loss:.4f}, train steps/sec: {steps_per_sec:.2f}")
                log_memory_usage(logger)

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


def log_memory_usage(logger):
    """Log current and maximum memory usage. Useful for debugging memory."""
    logger.debug(f"(memory usage) current: {bytes_to_gb(torch.cuda.memory_allocated()):.2f} GB, max: {bytes_to_gb(torch.cuda.max_memory_allocated()):.2f} GB")


def bytes_to_gb(n_bytes):
    return n_bytes / (1024 ** 3)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--base-model", type=str, default="stabilityai/stable-diffusion-2-1")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--results-dir", type=str, default="sd_results")
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--global-batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--ckpt-every", type=int, default=4000)
    args = parser.parse_args()
    main(args)
