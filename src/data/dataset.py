import torch
from torch.utils.data import Dataset


class Dataset(Dataset):
    def __init__(self, data, block_size) -> None:
        super().__init__()
        self.data = data
        self.block_size = block_size

    def __len__(self) -> int:
        # return len(self.data) - self.block_size - 1
        return len(self.data) // 10 - self.block_size - 1

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        array = self.data[index : index + self.block_size + 1]
        return array[:-1], array[1:]
