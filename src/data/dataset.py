import sys

import torch
from torch.utils.data import Dataset


# TODO: transform from iter to epochs
class Dataset(Dataset):
    def __init__(self, data, block_size) -> None:
        super().__init__()
        self.data = data
        self.block_size = block_size

    def __len__(self) -> int:
        return sys.maxsize

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        # For now I use this ad-hoc solution to leverage torch Dataset class
        idx = torch.randint(len(self.data) - self.block_size, (1,))
        x = self.data[idx : idx + self.block_size]
        y = self.data[idx + 1 : idx + self.block_size + 1]
        return x, y
