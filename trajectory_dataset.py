import os
import torch
from glob import glob
from torch.utils.data import Dataset


TRAJECTORY_FILENAME = "trajectory.pt"


class TrajectoryDataset(Dataset):
    def __init__(self, dir: str):
        self.trajectory_dirs = glob(os.path.join(dir, "*/"))
        assert len(self.trajectory_dirs) > 0, "no trajectories found in directory"

        self.num_timesteps = torch.load(self.get_trajectory_file(0), weights_only=True).shape[0]

    def __len__(self):
        return len(self.trajectory_dirs) * self.num_timesteps

    def get_trajectory_file(self, idx):
        return os.path.join(self.trajectory_dirs[idx], TRAJECTORY_FILENAME)

    def get_other_files(self, idx):
        files = glob(os.path.join(self.trajectory_dirs[idx], "*.pt"))
        return [file for file in files if file.split("/")[-1] != TRAJECTORY_FILENAME]

    def __getitem__(self, idx):
        trajectory_idx = idx // self.num_timesteps
        timestep = idx % self.num_timesteps

        # Get model in- and output
        trajectory = torch.load(self.get_trajectory_file(trajectory_idx), weights_only=True)
        model_input, model_output = trajectory[timestep]

        # Get other files that are provided by the dataset
        other_data = {}
        for file in self.get_other_files(trajectory_idx):
            name = file.split("/")[-1].split(".")[0]
            other_data[name] = torch.load(file, weights_only=True)

        return model_input, model_output, timestep + 1, other_data
