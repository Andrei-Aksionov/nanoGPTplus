from pathlib import Path

import torch
from loguru import logger
from torch.utils.data import DataLoader

from src import config
from src.data.dataset import Dataset
from src.data.tokenizer import CharTokenizer
from src.model.bigram import BigramLanguageModel
from src.model.trainer import Trainer
from src.utils.data import train_test_split
from src.utils.seed import set_seed


def train():

    set_seed(config.model.seed)

    logger.debug("Loading the data...")
    data_path = Path.cwd() / config.datasets.tiny_shakespeare.file_path
    with open(data_path, "r") as fin:
        text = fin.read()
    logger.debug("Data is loaded.")

    logger.debug("Starting tokenizing...")
    tokenizer = CharTokenizer(text)
    data = torch.tensor(tokenizer.encode(text))
    logger.debug("Tokenizing is done.")
    train_data, test_data = train_test_split(data, 0.9)

    logger.debug("Preparing data loaders...")
    model_config = config.model.small
    block_size = model_config.block_size
    batch_size = model_config.batch_size
    train_dataset = Dataset(train_data, block_size)
    test_dataset = Dataset(test_data, block_size)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, num_workers=1)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, num_workers=1)
    logger.debug("Data loaders are prepared.")

    logger.debug("Staring training...")
    model = BigramLanguageModel(vocab_size=tokenizer.vocab_size)
    optimizer = torch.optim.AdamW(model.parameters(), lr=model_config.learning_rate)
    trainer = Trainer(model, optimizer, train_dataloader, test_dataloader)
    trainer.train(epochs=model_config.epochs)
    logger.debug("Training is finished")


if __name__ == "__main__":
    train()
