import math
from typing import Optional

import torch


class CosineWarmupLRSchedular:
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_iters: int,
        lr_decay_iters: int,
        min_lr: Optional[float] = None,
    ) -> None:
        """Cosine learning rate schedular with warmup.

        Parameters
        ----------
        optimizer : torch.optim.Optimizer
            to what optimizer apply schedular
        warmup_iters : int
            for how long learning rate will be linearly increasing from min_lr to learning_rate
        lr_decay_iters : int
            for how long learning rate will be linearly decreasing after warmup is finished
        min_lr : Optional[float], optional
            this learning rate will be applied after lr decay is finished, by default None
        """
        self.optimizer = optimizer
        self.warmup_iters = warmup_iters
        self.lr_decay_iters = lr_decay_iters
        self.learning_rate = self.optimizer.param_groups[0]["lr"]
        self.min_lr = min_lr or self.learning_rate / 10

    def _get_lr(self, iteration: int) -> float:
        # linearly increase lr during warmup up to learning_rate
        if iteration < self.warmup_iters:
            return self.learning_rate * iteration / self.warmup_iters
        # when lr decay is finished -> set min_lr
        if iteration > self.lr_decay_iters:
            return self.min_lr
        # during lr decay (after warmup) use cosine decay from learning_rate down to min_lr
        decay_ratio = (iteration - self.warmup_iters) / (self.lr_decay_iters - self.warmup_iters)
        if not (0 <= decay_ratio <= 1):
            msg = f"Decay ratio has to be in range [0, 1], but it's actually equal to {decay_ratio}"
            raise ValueError(msg)
        coefficient = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coefficient ranges 0..1
        return self.min_lr + coefficient * (self.learning_rate - self.min_lr)

    def step(self, iteration: int) -> None:
        """Apply cosine learning rate decay with warmup.

        Parameters
        ----------
        iteration : int
            current iteration, affects whether warmup is applied or cosine lr decay
        """
        lr = self._get_lr(iteration)
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr
