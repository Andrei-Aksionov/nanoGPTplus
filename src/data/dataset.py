from math import ceil
from typing import Optional, Tuple

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset


class NextTokenDataset(Dataset):
    def __init__(self, data: np.ndarray, block_size: int, fraction: Optional[float] = None) -> None:
        """Create custom torch Dataset that returns inputs and targets.

        Targets are essentially the same as inputs, but shifted by one element to the right.
        It is done because we want to predict the next element in the sequence.
        For example:
        ```
            inputs =  [1, 2, 3, 4]
            targets = [2, 3, 4, 5]
        ```
        That means that when we have input `1` we want to predict `2`, when input is `2` - predict `3` and so on.

        Parameters
        ----------
        data : np.ndarray
            array from where pairs of inputs and targets will be generated
        block_size : int
            the length of a sequence that the model will process
        fraction: Optional[float]
            for debugging purposes one might want to use only a small fraction of the dataset
        """
        super().__init__()

        self.data = data
        self.block_size = block_size
        self.fraction = fraction

    def __len__(self) -> int:
        size = len(self.data) - self.block_size - 1
        return ceil(size * self.fraction) if self.fraction else size

    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor]:
        chunk = self.data[index : index + self.block_size + 1]
        return chunk[:-1], chunk[1:]


class NextTokenRandomDataset(Dataset):
    def __init__(self, data: np.ndarray, block_size: int, max_iter: int) -> None:
        """Create custom torch Dataset that returns inputs and targets by random sampling with replacement.

        Targets are essentially the same as inputs, but shifted by one element to the right.
        It is done because we want to predict the next element in the sequence.
        For example:
        ```
            inputs =  [1, 2, 3, 4]
            targets = [2, 3, 4, 5]
        ```
        That means that when we have input `1` we want to predict `2`, when input is `2` - predict `3` and so on.

        Parameters
        ----------
        data : np.ndarray
            array from where pairs of inputs and targets will be generated
        block_size : int
            the length of a sequence that the model will process
        max_iter : int
            how many sample should be randomly sampled till iterator is exhausted
        """
        super().__init__()
        self.data = data
        self.block_size = block_size
        self.max_iter = max_iter

    def __len__(self) -> int:
        return self.max_iter

    def __getitem__(self, _: int) -> Tuple[Tensor, Tensor]:
        # instead of provided index it will be randomly sampled
        index = torch.randint(len(self.data) - self.block_size - 1, (1,))
        chunk = self.data[index : index + self.block_size + 1]
        return chunk[:-1], chunk[1:]
