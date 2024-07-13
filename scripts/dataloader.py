from typing import Union

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

    _NUM_ELECTRODES = 10
    _EEG_INDEX = 2

    def __init__(self, samples):
        self.num_samples = len(samples)
        self.data = samples

        self.data_min = min([x[0][self._EEG_INDEX].min() for x in self.data])
        self.data_max = max([x[0][self._EEG_INDEX].max() for x in self.data])

    def __len__(self):
        return self.num_samples * self._NUM_ELECTRODES
        
    def __getitem__(self, idx):
        real_idx = idx // self._NUM_ELECTRODES
        idx_within = idx % self._NUM_ELECTRODES

        # Extract only the EEG signal
        sample = self.data[real_idx][0][self._EEG_INDEX].T[idx_within]

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

def MyDataLoader(train_file, test_file, batch_size, num_workers=1):
    print("----Loading dataset----")
    
    training = torch.load(train_file)  # Loads an object saved with torch.save() from a file
    validation = torch.load(test_file)  # Loads an object saved with torch.save() from a file
    
    train_dataset = MyDataset(training)
    eval_dataset = MyDataset(validation)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    eval_loader = torch.utils.data.DataLoader(eval_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    y_train = [y for x, y in training]
    _, train_distr = np.unique(y_train, return_counts=True) # number of labels in train dataset, for each class
    weights = sum(train_distr)/train_distr
    sample_weights= weights/sum(weights)  # sample_weights in case of unbalanced data

    print('Dataset: MAHNOB-HCI')
    print("#Traning samples: ", len(train_dataset))
    print("#Validation samples: ", len(eval_dataset))
    print("#Training distribution: ", train_distr)
    print("-------------------------")

    return train_loader, eval_loader, sample_weights