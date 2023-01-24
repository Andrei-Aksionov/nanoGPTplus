import torch
from tqdm import tqdm


class Trainer:
    def __init__(
        self,
        model,
        optimizer,
        train_dataloader,
        test_dataloader,
        eval_interval: int,
        eval_iters: int,
        loss=None,
    ) -> None:
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.train_dataloader = iter(train_dataloader)
        self.test_dataloader = iter(test_dataloader)
        self.eval_interval = eval_interval
        self.eval_iters = eval_iters
        self.loss = loss

    def train(self, num_iter: int):

        # TODO: transform from iter to epochs

        loop = tqdm(range(num_iter), ascii=True)

        self.model.train()
        for iter in loop:

            xb, yb = next(self.train_dataloader)

            _, loss = self.model(xb, yb)

            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)

            if iter % self.eval_interval == 0:
                self.model.eval()
                with torch.no_grad():
                    test_loss_sum = 0
                    for _ in range(self.eval_iters):
                        x_test, y_test = next(self.test_dataloader)
                        _, loss = self.model(x_test, y_test)
                        test_loss_sum += loss.item()
                    test_loss = test_loss_sum / self.eval_iters

            loop.set_postfix(loss=loss.item(), test_loss=test_loss)

        self.final_loss = loss.item()
