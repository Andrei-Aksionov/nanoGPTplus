import torch


class Dataset:
    def __init__(self, data, block_size, batch_size) -> None:
        super().__init__()
        self.data = data
        self.block_size = block_size
        self.batch_size = batch_size

    # TODO: get rid of mode
    def get_batch(self, mode):
        # generate random indices
        indices = torch.randint(len(self.data) - self.block_size, (self.batch_size,))

        # generate batches
        x = torch.stack([self.data[idx : idx + self.block_size] for idx in indices])
        # targets are inputs with shift of 1 - predict the next character
        y = torch.stack([self.data[idx + 1 : idx + self.block_size + 1] for idx in indices])

        return x, y
