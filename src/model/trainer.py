import torch
import torch.nn.functional as F
from loguru import logger
from torch import Tensor, nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.utils.model import save_checkpoint


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        train_dataloader: DataLoader,
        eval_dataloader: DataLoader,
        device: torch.device | None,
        loss: "torch.nn.modules" = None,
        tqdm_update_interval: int = 100,
        checkpoint_model_path: str = "models/model.pth.tar",
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
        device: torch.device | None
            where the model and batch should be stored and executed
            if device is None, batch will be moved to the same device where the model is
        loss : torch.nn.modules, optional
            function to measure correctness of predictions, if not provided the model should contain it, by default None
        tqdm_update_interval : int, optional
            how often (in batches) loss value should be updated, by default 100
        checkpoint_model_path : str, optional
            where to save the best model, by default "models/model.pth.tar"

        Raises
        ------
        ValueError
            if loss-function is provided and it's not cross-entropy from torch.nn.functional
        ValueError
            if loss-function is not provided and the model instance doesn't contain such method
        """
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        if device:
            self.device = device
            self.model.to(self.device)
        else:
            self.device = next(model.parameters()).device
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
        self.checkpoint_model_path = checkpoint_model_path

    def __move_batch_to(self, batch: list[Tensor, Tensor]) -> list[Tensor, Tensor]:
        return [x.to(self.device) for x in batch]

    def train(self, epochs: int) -> None:
        """Train the model for specified number of epochs.

        Parameters
        ----------
        epochs : int
            the number of epochs the model will be training
        """
        logger.debug("Training on '{}' device".format(self.device))

        best_eval_loss = float("inf")

        for epoch in range(epochs):
            tqdm.write(f" Epoch: {epoch} ".center(40, "="))

            # reuse code for training and evaluation
            for mode, dataloader in zip(
                ["train", "eval"],
                [self.train_dataloader, self.eval_dataloader],
            ):
                self.model.train() if mode == "train" else self.model.eval()

                loop = tqdm(dataloader, desc=mode, ascii=True)

                loss_batch = torch.tensor(0.0, device=self.device)
                eval_loss = torch.tensor(0.0, device=self.device)
                for idx, batch in enumerate(loop):
                    inputs, targets = self.__move_batch_to(batch)

                    # during evaluation there is no need to store any information for backpropagation
                    with torch.set_grad_enabled(mode == "train"):
                        logits = self.model(inputs)
                        loss = self.loss(logits, targets)

                    # if training -> do the gradient descent
                    if mode == "train":
                        loss.backward()
                        self.optimizer.step()
                        self.optimizer.zero_grad()

                    with torch.no_grad():
                        loss_batch += loss
                        if mode == "eval":
                            eval_loss += loss

                    # update loss value in the tqdm output every `n` batches, so it's not updated too frequently
                    if idx and idx % self.tqdm_update_interval == 0:
                        loop.set_postfix(loss=loss_batch.item() / self.tqdm_update_interval)
                        loss_batch.zero_()

            eval_loss /= len(self.eval_dataloader)
            tqdm.write(f"Eval averaged loss: {eval_loss}")
            if eval_loss < best_eval_loss:
                logger.info(
                    "Current eval loss is `{:.4f}` which is smaller than current best loss is `{:.4f}`; "
                    "saving the model...".format(eval_loss, best_eval_loss),
                )
                best_eval_loss = eval_loss
                save_checkpoint(state=self.model.state_dict(), path=self.checkpoint_model_path)
                logger.info("Best model is saved.")
