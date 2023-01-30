import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.utils.device import get_device


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        train_dataloader: DataLoader,
        eval_dataloader: DataLoader,
        loss: "torch.nn.modules" = None,
        tqdm_update_interval: int = 100,
    ) -> None:
        """Contains boilerplate to train and evaluate the model.

        Parameters
        ----------
        model : nn.Module
            the model that will be trained
        optimizer : torch.optim.Optimizer
            optimizer to update the weights
        train_dataloader : DataLoader
            dataloader containing data for training
        eval_dataloader : DataLoader
            dataloader containing data for evaluation
        loss : torch.nn.modules, optional
            function to measure correctness of predictions, if not provided the model should contain it, by default None
        tqdm_update_interval : int, optional
            how often (in batches) loss value should be updated, by default 100

        Raises
        ------
        ValueError
            _description_
        """
        super().__init__()
        self.model = model.to(get_device())
        self.optimizer = optimizer
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        # either model should contain the loss or the loss function has to be provided
        if loss:
            if loss is not F.cross_entropy:
                raise ValueError("So far only cross-entropy is supported")
            self.loss = loss
        else:
            if not hasattr(self.model, "loss"):
                raise ValueError("Loss is not provided and model instance doesn't have such method")
            self.loss = self.model.loss
        self.tqdm_update_interval = tqdm_update_interval

    def __move_batch_to(
        self,
        batch: list[Tensor, Tensor],
        device: torch.device = None,
    ) -> list[Tensor, Tensor]:
        if not device:
            device = next(self.model.parameters()).device
        return [x.to(device) for x in batch]

    def train(self, epochs: int) -> None:
        """Train the model for specified number of epochs.

        Parameters
        ----------
        epochs : int
            the number of epochs the model will be training
        """
        for epoch in range(epochs):

            tqdm.write(f" Epoch: {epoch} ".center(40, "="))

            # reuse code for training and evaluation
            for mode, dataloader in zip(
                ["train", "eval"],
                [self.train_dataloader, self.eval_dataloader],
            ):

                if mode == "train":
                    self.model.train()
                else:
                    self.model.eval()

                loop = tqdm(dataloader, desc=mode, ascii=True)

                loss_accumulated = torch.tensor(0.0)
                for idx, batch in enumerate(loop):
                    inputs, targets = self.__move_batch_to(batch)

                    # if evaluation there is no need to store any information for backpropagation
                    with torch.set_grad_enabled(mode == "train"):
                        logits = self.model(inputs)
                        loss = self.loss(logits, targets)

                    # if training -> do the gradient descent
                    if mode == "train":
                        loss.backward()
                        self.optimizer.step()
                        self.optimizer.zero_grad()

                    with torch.no_grad():
                        loss_accumulated += loss

                    # update loss value in the tqdm output only `n` batches, so it's not flashing
                    if idx % self.tqdm_update_interval == 0:
                        loop.set_postfix(loss=loss_accumulated.item() / self.tqdm_update_interval)
                        loss_accumulated.zero_()
