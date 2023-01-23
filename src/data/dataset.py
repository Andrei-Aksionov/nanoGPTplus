from pathlib import Path

import torch
from torch.utils.data import Dataset

from src import config


# TODO: this whole code need a major refactoring
# class Dataset(Dataset):
class Dataset:
    def __init__(self) -> None:

        # read data
        # TODO: filename or even the whole path should be stored inside config file
        data_path = (
            Path(__file__).parents[2]
            / config.datasets.tiny_shakespeare.folder
            / Path(config.datasets.tiny_shakespeare.url).name
        )

        with open(data_path, "r") as fin:
            dataset = fin.read()

        # get all unique characters and count them
        self.characters = sorted(set(dataset))
        self.vocab_size = len(self.characters)

        # create mapping from characters to integers and inverse
        self.stoi = {char: idx for idx, char in enumerate(self.characters)}
        self.itos = dict(enumerate(self.characters))

        # train and test splits
        self.data = torch.tensor(self.encode(dataset), dtype=torch.long)
        test_split = int(len(self.data) * 0.9)
        print(f"Test split: {test_split}")
        self.train_data = self.data[:test_split]
        self.test_data = self.data[test_split:]

        self.block_size = config.model.small.block_size
        self.batch_size = config.model.small.batch_size

    def encode(self, text: str) -> list[int]:
        return [self.stoi[ch] for ch in text]

    def decode(self, indices: list[int]) -> str:
        return "".join(self.itos[idx] for idx in indices)

    def get_batch(self, mode: str):
        # mode: train or test
        assert mode in {"train", "test"}, "Only support train or test"

        data = self.train_data if mode == "train" else self.test_data

        # generate random indices
        indices = torch.randint(len(data) - self.block_size, (self.batch_size,))

        # generate batches
        x = torch.stack([data[idx : idx + self.block_size] for idx in indices])
        # targets are inputs with shift of 1 - predict the next character
        y = torch.stack([data[idx + 1 : idx + self.block_size + 1] for idx in indices])

        return x, y
