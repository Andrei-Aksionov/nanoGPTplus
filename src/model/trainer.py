from typing import List, Optional, Union

import torch
from loguru import logger
from torch import Tensor, nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.utils import save_checkpoint


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        train_dataloader: DataLoader,
        eval_dataloader: DataLoader,
        device: Union[None, str, torch.device],
        lr_schedular: Optional[torch.optim.lr_scheduler.Optimizer] = None,
        loss: "torch.nn.modules" = None,
        grad_accumulation_steps: Optional[int] = None,
        clip_grad_norm: Optional[float] = 1.0,
        checkpoint_model_path: str = "models/model.pth.tar",
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
        device: Union[None, str, torch.device]
            where the model and batch should be stored and executed
            if device is None, batch will be moved to the same device where the model is
        lr_schedular: Optional[torch.optim.lr_schedular.Optimizer]
            learning rate schedular, by default None
        loss : torch.nn.modules, optional
            function to measure correctness of predictions, if not provided the model should contain it, by default None
        grad_accumulation_steps: None | int
            if provided gradients will be zeroed every n steps
        clip_grad_norm: Optional[float]
            clips gradient norm of an iterable of parameters. The norm is computed over all gradients together, as if
            they were concatenated into a single vector. Gradients are modified in-place, by default 1.0
        checkpoint_model_path : str, optional
            where to save the best model, by default "models/model.pth.tar"
        tqdm_update_interval : int, optional
            how often (in batches) loss value should be updated, by default 100

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
        self.lr_schedular = lr_schedular
        # either model should contain the loss or the loss function has to be provided
        if loss:
            self.loss = loss
        elif hasattr(self.model, "loss"):
            self.loss = self.model.loss
        else:
            raise ValueError("Loss is not provided and model instance doesn't have such method")
        self.grad_accumulation_steps = grad_accumulation_steps
        self.clip_grad_norm = clip_grad_norm
        self.tqdm_update_interval = tqdm_update_interval
        self.checkpoint_model_path = checkpoint_model_path

    def __move_batch_to(self, batch: List[Tensor]) -> List[Tensor]:
        return [x.to(self.device) for x in batch]

    def _train_step(
        self,
        mode: str,
        idx: int,
        batch: Union[tuple, list],
    ) -> Tensor:
        # data should be on the same device as the model
        inputs, targets = self.__move_batch_to(batch)
        # during evaluation there is no need to store any information for backpropagation
        with torch.set_grad_enabled(mode == "train"):
            logits = self.model(inputs)
            loss = self.loss(logits, targets)
        # if training -> do the gradient descent
        if mode == "train":
            loss.backward()
            if self.clip_grad_norm:
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.clip_grad_norm)
            # do weight update only every n grad accumulation steps if provided or every step if not
            if not self.grad_accumulation_steps or idx % self.grad_accumulation_steps == 0:
                self.optimizer.step()
                if self.lr_schedular:
                    self.lr_schedular.step(idx)
                self.optimizer.zero_grad(set_to_none=True)
        return loss

    def train(self, epochs: int) -> None:
        """Train the model for specified number of epochs.

        Parameters
        ----------
        epochs : int
            the number of epochs the model will be training
        """
        logger.debug("Training on '{}' device".format(self.device))
        best_eval_loss = float("inf")
        # iterate over epochs
        for epoch in range(epochs):
            tqdm.write(f" Epoch: {epoch} ".center(40, "="))
            # reuse code for training and evaluation
            for mode, dataloader in zip(
                ["train", "eval"],
                [self.train_dataloader, self.eval_dataloader],
            ):
                # set model into train or eval mode: required for BatchNorm or Dropout
                self.model.train() if mode == "train" else self.model.eval()
                tqdm_loop = tqdm(dataloader, desc=mode, ascii=True)
                epoch_loss = torch.tensor(0.0, device=self.device)  # to store accumulated loss during an epoch
                # iterate over batches
                for idx, batch in enumerate(tqdm_loop, start=1):
                    loss = self._train_step(mode, idx, batch)
                    # do not add `epoch_loss` into the computational graph
                    with torch.no_grad():
                        epoch_loss += loss
                    # update loss value in the tqdm output every `n` batches, so it's not updated too frequently
                    if idx % self.tqdm_update_interval == 0:
                        tqdm_loop.set_postfix(loss=epoch_loss.item() / idx)

                # save best performing model (epoch with the smallest eval loss)
                if mode == "eval":
                    eval_loss = epoch_loss.item() / len(dataloader)
                    tqdm.write(f"Eval averaged loss: {eval_loss:.4f}")
                    if eval_loss < best_eval_loss:
                        logger.info(
                            "Current eval loss is `{:.4f}` which is smaller than current best loss of `{:.4f}`; "
                            "saving the model...".format(eval_loss, best_eval_loss),
                        )
                        best_eval_loss = eval_loss
                        save_checkpoint(state=self.model.state_dict(), path=self.checkpoint_model_path)
                        logger.info("Best model is saved.")
