import sys

import torch
import torch.nn.functional as F
from tqdm import tqdm

from src.utils.device import get_device


class Trainer:
    def __init__(
        self, model, optimizer, train_dataloader, test_dataloader, loss=None, tqdm_update_interval: int = 100
    ) -> None:
        super().__init__()
        self.model = model.to(get_device())
        self.optimizer = optimizer
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        if loss:
            assert loss is F.cross_entropy, "So far only cross-entropy is supported"
            self.loss = loss
        else:
            if not hasattr(self.model, "loss"):
                raise ValueError("Loss is not provided and model instance doesn't have such method")
            else:
                self.loss = self.model.loss
        self.tqdm_update_interval = tqdm_update_interval

    def move_batch_to(self, batch, device: torch.device = None):
        if not device:
            device = next(self.model.parameters()).device
        return [x.to(device) for x in batch]

    def train(self, epochs: int):

        for epoch in range(epochs):

            tqdm.write(f" Epoch: {epoch} ".center(40, "="))

            for mode, dataloader in zip(
                ["train", "eval"],
                [self.train_dataloader, self.test_dataloader],
            ):

                if mode == "train":
                    self.model.train()
                else:
                    self.model.eval()

                loop = tqdm(dataloader, desc=mode, ascii=True)

                loss_accumulated = torch.tensor(0.0)
                for idx, batch in enumerate(loop):
                    inputs, targets = self.move_batch_to(batch)

                    with torch.set_grad_enabled(mode == "train"):
                        logits = self.model(inputs)
                        loss = self.loss(logits, targets)

                    if mode == "train":
                        loss.backward()
                        self.optimizer.step()
                        self.optimizer.zero_grad()

                    with torch.no_grad():
                        loss_accumulated += loss

                    if idx % self.tqdm_update_interval == 0:
                        loop.set_postfix(loss=loss_accumulated.item() / 100)
                        loss_accumulated.zero_()
