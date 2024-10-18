import os
import csv
import torch
from torch.utils.data import Dataset


class SyntheticDiffusionSamplesDataset(Dataset):
    """Synthetic diffusion samples dataset that consists of trajectories from a pre-trained model.
    The dataset is a collection of trajectories saved as PyTorch tensors. Each trajectory is a sequence of
    latent vectors sampled from a pre-trained model at each timestep. Indexing into the dataset returns
    a pair of consecutive timesteps from the same trajectory.

    Expects a `{dir}/info.csv` file, where the first column contains the filenames of the trajectories
    and the rest of the columns define metadata about the trajectory. E.g.:
    ```
    trajectory,class_label,cfg_scale
    b2n1jp.pt,1,3.4
    as4j1d.pt,2,2.8
    vws2jk.pt,3,1.1
    ```

    Args:
        dir (str): The directory containing the trajectories.
        num_timesteps (int): The number of timesteps in each trajectory.

    Example:
    ```python
    dataset = SyntheticDiffusionSamplesDataset("trajectory_data")
    # `trajectory_info` is a dictionary containing metadata about the trajectory.
    prev, next, trajectory_info = dataset[idx]
    ```
    
    """

    def __init__(self, dir: str, num_timesteps: int=1000):
        self.dir = dir
        self.num_timesteps = num_timesteps

        self.trajectories = []
        info_csv = csv.reader(open(os.path.join(dir, "info.csv")))
        headers = next(info_csv)[1:]
        for row in info_csv:
            key, *values = row
            self.trajectories.append((key, dict(zip(headers, values))))

    def __len__(self):
        return len(self.trajectory_filenames * (self.num_timesteps - 1))

    def __getitem__(self, idx):
        trajectory_idx = idx // (self.num_timesteps - 1)
        timestep = self.num_timesteps - (idx % (self.num_timesteps - 1)) - 1

        trajectory_filename, info = self.trajectories[trajectory_idx]
        trajectory = torch.load(trajectory_filename)

        return trajectory[timestep - 1], trajectory[timestep], info
