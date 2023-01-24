import torch
import torch.nn.functional as F
from tqdm import tqdm


class Trainer:
    def __init__(
        self,
        model,
        optimizer,
        train_dataloader,
        test_dataloader,
        loss,
    ) -> None:
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        assert loss is F.cross_entropy, "So far only cross-entropy is supported"
        self.loss = loss

    def train(self, epochs: int):

        for epoch in range(epochs):

            print(f" Epoch: {epoch} ".center(40, "="))

            train_loop = tqdm(self.train_dataloader, ascii=True)
            self.model.train()
            batch_loss = torch.tensor(0.0)
            for idx, batch in enumerate(train_loop):
                # TODO: maybe not x and y, but inputs and targets?
                x, y = batch
                logits = self.model(x)

                B, T, C = logits.shape
                loss = self.loss(
                    logits.view(B * T, C),
                    y.view(B * T),
                )

                batch_loss += loss
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)

                if idx % 100 == 0:
                    train_loop.set_postfix(train_loss=batch_loss.item() / 100)
                    batch_loss = 0

            self.model.eval()
            test_loss = torch.tensor(0.0)
            test_loop = tqdm(self.test_dataloader, ascii=True)
            for idx, batch in enumerate(test_loop):
                x_test, y_test = batch
                logits = self.model(x_test)
                B, T, C = logits.shape
                loss = self.loss(
                    logits.view(B * T, C),
                    y_test.view(B * T),
                )
                test_loss += loss

                if idx % 100 == 0:
                    test_loop.set_postfix(test_loss=test_loss.item() / 100)
                    test_loss = 0

            train_loop.close()
            test_loop.close()
