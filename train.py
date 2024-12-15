import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader
from accelerate import Accelerator
from accelerate.utils import set_seed
from accelerate.logging import get_logger
from glob import glob
from time import time
import warnings
import argparse
import logging
import yaml
import os

from src.models import get_adapter_unet
from src.supported_models import TEXT_MODELS, SUPPORTED_MODELS


def main(args):
    assert args.base_model in SUPPORTED_MODELS, f"model {args.base_model} not supported"
    is_text_model = args.base_model in TEXT_MODELS

    assert torch.cuda.is_available(), "training requires at least one gpu"
    set_seed(args.seed)

    experiment_index = len(glob(f"{args.results_dir}/*"))
    experiment_dir = os.path.join(args.results_dir, f"{experiment_index:03d}")
    os.makedirs(experiment_dir, exist_ok=True)
    logger = create_logger(experiment_dir, verbose=args.verbose)

    # Save arguments
    with open(os.path.join(experiment_dir, "config.yaml"), "w") as f:
        yaml.dump(vars(args), f)

    # Set up accelerator
    with warnings.catch_warnings(action="ignore", category=FutureWarning):
        accelerator = Accelerator(project_dir=experiment_dir)

    logger.info(f"experiment directory created at {experiment_dir}")

    # Create model and only make adapters trainable
    model = get_adapter_unet(
        args.base_model,
        hidden_dim=args.hidden_dim,
        use_prompt_cond=args.use_prompt_cond,
        use_neg_prompt_cond=args.use_neg_prompt_cond,
    )
    model.train_adapters()

    # Optimizer
    opt = torch.optim.AdamW(model.adapter_parameters(), lr=1e-4, weight_decay=0)
    model.train()

    # Log number of parameters
    logger.info(f"model parameters: {sum(p.numel() for p in model.parameters()):,}")
    logger.info(f"adapter parameters: {sum(p.numel() for p in model.adapter_parameters()):,}")
    logger.info(f"requires_grad parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # Setup data
    dataset = TrajectoryDataset(args.data_path)
    loader = DataLoader(
        dataset,
        batch_size=int(args.global_batch_size // accelerator.num_processes),
        num_workers=args.num_workers,
        shuffle=True,
        drop_last=True,
    )
    logger.info(f"dataset contains {len(dataset):,} points ({args.data_path})")

    # Prepare model, optimizer, and loader
    loader, model, opt = accelerator.prepare(loader, model, opt)

    # Variables for monitoring/logging purposes
    train_steps = 0
    log_steps = 0
    running_loss = 0
    start_time = time()

    logger.info(f"training for {args.epochs} epochs...")

    for epoch in range(args.epochs):
        logger.info(f"beginning epoch {epoch}...")

        for teacher_input, teacher_output, timestep, condition in loader:
            cfg_scale = condition["cfg_scale"]
            prompt = condition.get("prompt", None)
            neg_prompt = condition.get("neg_prompt", None)
            class_label = condition.get("class_label", None)
            additional_model_kwargs = condition.get("additional_model_kwargs", dict())

            # Give conditioning variables to adapter blocks
            if isinstance(model, DDP):
                model.module.set_adapter_kwargs(
                    cfg_scale=cfg_scale,
                    prompt=prompt,
                    neg_prompt=neg_prompt,
                    class_label=class_label,
                )
            else:
                model.set_adapter_kwargs(
                    cfg_scale=cfg_scale,
                    prompt=prompt,
                    neg_prompt=neg_prompt,
                    class_label=class_label,
                )

            if is_text_model:
                model_output = model(
                    teacher_input,
                    timestep,
                    encoder_hidden_states=prompt,
                    **additional_model_kwargs,
                ).sample
            else:
                model_output = model(
                    teacher_input,
                    timestep,
                    class_labels=class_label,
                    **additional_model_kwargs,
                ).sample

            # For DiT, we have to remove the variance prediction
            loss = F.mse_loss(model_output[:, :teacher_output.shape[1]], teacher_output)

            # Backward pass
            accelerator.backward(loss)
            opt.step()
            opt.zero_grad()

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
                if accelerator.is_main_process:
                    logger.info(f"(step={train_steps:07d}) train loss: {avg_loss:.4f}, train steps/sec: {steps_per_sec:.2f}")
                    log_memory_usage(logger)

                # Reset monitoring variables
                running_loss = 0
                log_steps = 0
                start_time = time()

            # Save checkpoint
            if train_steps % args.ckpt_every == 0 and train_steps > 0 and accelerator.is_main_process:
                accelerator.save_state(os.path.join(experiment_dir, "checkpoints", f"{train_steps:07d}"))

    logger.info("done!")


class TrajectoryDataset(Dataset):
    def __init__(self, dir: str):
        self.trajectory_dirs = glob(os.path.join(dir, "*/"))
        assert len(self.trajectory_dirs) > 0, "no trajectories found in directory"

        self.timesteps = torch.load(
            os.path.join(dir, "timesteps.pt"),
            weights_only=True,
        )
        self.latent_shape = torch.load(
            os.path.join(dir, "latent_shape.pt"),
            weights_only=True,
        )
        self.num_steps = self.timesteps.shape[0]

    def __len__(self):
        return len(self.trajectory_dirs) * self.num_steps

    def __getitem__(self, idx):
        trajectory_idx = idx // self.num_steps
        step = idx % self.num_steps

        # Get model in- and output
        trajectory = np.memmap(
            os.path.join(self.trajectory_dirs[trajectory_idx], "trajectory.npy"),
            dtype=np.float32,
            mode="r",
            shape=(self.num_steps, 2, *self.latent_shape),
        )
        model_input, model_output = torch.from_numpy(trajectory[step].copy())

        # Get conditioning
        conditioning = torch.load(
            os.path.join(self.trajectory_dirs[trajectory_idx], "conditioning.pt"),
            weights_only=True,
        )

        return model_input, model_output, self.timesteps[step], conditioning


def create_logger(logging_dir: os.PathLike, verbose: bool=False) -> logging.Logger:
    """Create a logger that writes to a log file and stdout."""

    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="[\033[34m%(asctime)s\033[0m] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(), logging.FileHandler(os.path.join(logging_dir, "log.txt"))]
    )
    return get_logger(__name__)


def log_memory_usage(logger):
    """Log current and maximum memory usage. Useful for debugging memory."""

    logger.debug(f"(memory usage) current: {bytes_to_gb(torch.cuda.memory_allocated()):.2f} GB, max: {bytes_to_gb(torch.cuda.max_memory_allocated()):.2f} GB")


def bytes_to_gb(n_bytes):
    return n_bytes * 1e-9


if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn", force=True)

    parser = argparse.ArgumentParser()

    # Training loop and model definition
    parser.add_argument("--results-dir", type=str, required=True)
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--base-model", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--global-batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--ckpt-every", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--verbose", action="store_true")

    # Adapter configuration
    parser.add_argument("--hidden-dim", type=int, default=320)
    parser.add_argument("--use-prompt-cond", action="store_true")
    parser.add_argument("--use-neg-prompt-cond", action="store_true")

    args = parser.parse_args()
    main(args)
