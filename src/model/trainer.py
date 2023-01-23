from tqdm import tqdm


class Trainer:
    def __init__(self, model, optimizer, dataset, loss=None) -> None:
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.loss = loss
        self.dataset = dataset

    def train(self, num_iter: int):

        loop = tqdm(range(num_iter), ascii=True)

        for _ in loop:

            xb, yb = self.dataset.get_batch(mode="train")

            logits, loss = self.model(xb, yb)

            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)

            loop.set_postfix(loss=loss.item())

        self.final_loss = loss.item()
