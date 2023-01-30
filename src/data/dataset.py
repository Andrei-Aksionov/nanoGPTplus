import numpy as np
from torch import Tensor
from torch.utils.data import Dataset


class Dataset(Dataset):
    def __init__(self, data: np.ndarray, block_size: int) -> None:
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
        """
        super().__init__()

        self.data = data
        self.block_size = block_size

    def __len__(self) -> int:
        return len(self.data) - self.block_size - 1

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        array = self.data[index : index + self.block_size + 1]
        return array[:-1], array[1:]
