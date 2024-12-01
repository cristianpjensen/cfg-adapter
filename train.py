import torch
import torch.nn.functional as F
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


# TODO: Negative prompting


def main(args):
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
    model = get_adapter_unet(args.base_model)(
        hidden_dim=args.hidden_dim,
        use_prompt_cond=args.use_prompt_cond,
        use_neg_prompt_cond=args.use_neg_prompt_cond,
        use_block_query=args.use_block_query,
    )
    model.train_adapters()
    trainable_params = list(filter(lambda p: p.requires_grad, model.parameters()))
    opt = torch.optim.AdamW(trainable_params, lr=1e-4, weight_decay=0)

    # Log number of parameters
    num_model_params = sum(p.numel() for p in model.parameters())
    num_adapter_params = sum(p.numel() for p in model.adapters.parameters()) - sum(p.numel() for p in model._blocks_with_adapters.parameters())
    num_trainable_model_params = sum(p.numel() for p in trainable_params)

    logger.info(f"total model parameters: {num_model_params:,}")
    logger.info(f"adapter parameters: {num_adapter_params:,}")
    logger.info(f"trainable model parameters: {num_trainable_model_params:,}")
    
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
    loader, model, opt = accelerator.prepare(loader, model, opt)

    # Variables for monitoring/logging purposes
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

            model.set_adapter_kwargs(
                cfg_scale=adapter_kwargs["cfg_scale"],
                prompt=model_kwargs["encoder_hidden_states"],
            )
            model_output = model(teacher_input, timestep, *model_args, **model_kwargs).sample
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

    logger.info("done!")


class TrajectoryDataset(Dataset):
    def __init__(self, dir: str):
        self.trajectory_dirs = glob(os.path.join(dir, "*/"))
        assert len(self.trajectory_dirs) > 0, "no trajectories found in directory"

        self.num_steps = torch.load(self.get_trajectory_file(0), weights_only=True).shape[0]
        self.timesteps = torch.load(os.path.join(dir, "timesteps.pt"), weights_only=True)

    def __len__(self):
        return len(self.trajectory_dirs) * self.num_steps

    def get_trajectory_file(self, idx):
        return os.path.join(self.trajectory_dirs[idx], "trajectory.pt")

    def get_other_files(self, idx):
        files = glob(os.path.join(self.trajectory_dirs[idx], "*.pt"))
        return [file for file in files if file.split("/")[-1] != "trajectory.pt"]

    def __getitem__(self, idx):
        trajectory_idx = idx // self.num_steps
        step = idx % self.num_steps

        # Get model in- and output
        trajectory = torch.load(self.get_trajectory_file(trajectory_idx), weights_only=True)
        model_input, model_output = trajectory[step]

        # Get other files that are provided by the dataset
        other_data = {}
        for file in self.get_other_files(trajectory_idx):
            name = file.split("/")[-1].split(".")[0]
            other_data[name] = torch.load(file, weights_only=True)

        return model_input, model_output, self.timesteps[step], other_data


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
    parser = argparse.ArgumentParser()

    # Training loop and model definition
    parser.add_argument("--results-dir", type=str, required=True)
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--base-model", type=str, default="stabilityai/stable-diffusion-2-1")
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
    parser.add_argument("--use-block-query", action="store_true")

    args = parser.parse_args()
    main(args)
