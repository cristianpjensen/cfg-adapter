import os
import csv
import torch
from torch.utils.data import Dataset


class SyntheticDiffusionSamplesDataset(Dataset):
    """Expects a `{dir}/info.csv` file, where the first column contains the filenames of the
    trajectory samples and the rest of the columns define metadata about the trajectory. E.g.:
    ```
    sample_pred_filename,timestep,class_label,cfg_scale
    b2n1jp.pt,999,1,3.4
    as4j1d.pt,998,2,2.8
    vws2jk.pt,997,3,1.1
    ```
    The metadata should be readable as floats and will be read as such.

    The file is expected to contain a tensor of shape [2, C, H, W], where the first index is the
    input tensor and the second index is the output tensor of the teacher model.

    Args:
        dir (str): The directory containing the trajectories.

    Example:
    ```python
    dataset = SyntheticDiffusionSamplesDataset("trajectory_data")
    # `trajectory_info` is a dictionary containing metadata about the trajectory.
    model_input, model_output, trajectory_info = dataset[idx]
    ```
    
    """

    def __init__(self, dir: str):
        self.dir = dir

        # Read in sample information from info.csv
        self.samples = []
        info_csv = csv.reader(open(os.path.join(dir, "info.csv")))
        headers = next(info_csv)[1:]
        for row in info_csv:
            key, *values = row
            self.samples.append((key, dict(zip(headers, map(float, values)))))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        filename, info = self.samples[idx]
        sample = torch.load(os.path.join(self.dir, filename), weights_only=True)
        model_input, model_output = sample
        return model_input, model_output, info
