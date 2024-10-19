import os
import csv
import torch
from torch.utils.data import Dataset


class SyntheticDiffusionSamplesDataset(Dataset):
    """Expects a `{dir}/info.csv` file, where the first column contains the filenames of the
    trajectory samples and the rest of the columns define metadata about the trajectory. E.g.:
    ```
    trajectory_filename,cfg_scale,class_label
    b2n1jp.pt,3.4,0
    as4j1d.pt,2.8,1
    vws2jk.pt,1.1,2
    ...
    ```
    The metadata should be readable as floats and will be read as such.

    The file is expected to contain a tensor of shape [T, 2, C, H, W], where the first index of the
    second dimension is the input tensor and the second index is the output tensor of the teacher
    model (i.e., predicted noise).

    Args:
        dir (str): The directory containing the trajectories.

    Example:
    ```python
    dataset = SyntheticDiffusionSamplesDataset("trajectory_data")
    # `trajectory_info` is a dictionary containing metadata about the trajectory.
    model_input, model_output, trajectory_info = dataset[idx]
    ```
    
    """

    def __init__(self, dir: str, num_timesteps: int):
        self.dir = dir
        self.num_timesteps = num_timesteps

        # Read in trajectory information from info.csv
        self.trajectories = []
        info_csv = csv.reader(open(os.path.join(dir, "info.csv")))
        metadata_headers = next(info_csv)[1:]
        for row in info_csv:
            key, *metadata = row
            self.trajectories.append((key, dict(zip(metadata_headers, map(float, metadata)))))

    def __len__(self):
        return len(self.trajectories) * self.num_timesteps

    def __getitem__(self, idx):
        trajectory_idx = idx // self.num_timesteps
        timestep = idx % self.num_timesteps

        filename, metadata = self.trajectories[trajectory_idx]
        trajectory = torch.load(os.path.join(self.dir, filename), weights_only=True)
        model_input, model_output = trajectory[timestep]

        return model_input, model_output, timestep, metadata
