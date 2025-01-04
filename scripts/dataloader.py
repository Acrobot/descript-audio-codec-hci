from typing import Union, Literal

from audiotools import AudioSignal
from audiotools.core import util

import numpy as np
import torch

class MyDataset(torch.utils.data.Dataset):
    """
    Class to store a given dataset.
    Parameters:
    - samples: list of lists, each containing list of tensors (modalities) + labels
    """

    _CONFIGS = {
        "eye": {"index": 0, "channels": 4},
        "gsr": {"index": 1, "channels": 1},
        "eeg": {"index": 2, "channels": 10},
        "ecg": {"index": 3, "channels": 3},
    }

    def __init__(self, samples, config: Literal["eye", "gsr", "eeg", "ecg"]):
        self.num_samples = len(samples)
        self.data = samples

        self.index = self._CONFIGS[config]["index"]
        self.channels = self._CONFIGS[config]["channels"]

        self.data_min = min([x[0][self.index].min() for x in self.data])
        self.data_max = max([x[0][self.index].max() for x in self.data])

    def __len__(self):
        return self.num_samples * self.channels
        
    def __getitem__(self, idx):
        real_idx = idx // self.channels
        idx_within = idx % self.channels

        # Extract only the right signal
        sample = self.data[real_idx][0][self.index]

        if self.channels != 1:
            sample = sample.T[idx_within]

        # Try normalising the data
        sample = (sample - self.data_min) / (self.data_max - self.data_min)

        item = {
            "signal": AudioSignal(sample, sample_rate=256),
            "source_idx": real_idx,
            "item_idx": idx_within,
            "source": "Dataset",
            "path": f"{real_idx}/{idx_within}",
        }
        return item

    @staticmethod
    def collate(list_of_dicts: Union[list, dict], n_splits: int = None):
        """Collates items drawn from this dataset. Uses
        :py:func:`audiotools.core.util.collate`.

        Parameters
        ----------
        list_of_dicts : typing.Union[list, dict]
            Data drawn from each item.
        n_splits : int
            Number of splits to make when creating the batches (split into
            sub-batches). Useful for things like gradient accumulation.

        Returns
        -------
        dict
            Dictionary of batched data.
        """
        return util.collate(list_of_dicts, n_splits=n_splits)