from pathlib import Path

import torch
from loguru import logger
from torch.utils.data import DataLoader

from src import config
from src.data.dataset import NextTokenDataset, NextTokenRandomDataset
from src.data.tokenizer import CharTokenizer
from src.model.gpt_lm.gpt import GPTLanguageModel
from src.model.trainer import Trainer
from src.utils.data import train_test_split
from src.utils.device import get_device
from src.utils.seed import set_seed


def train() -> None:
    """Train bigram language model.

    Performs 4 steps:
    1. Loads the data
    2. Create tokenizer
    3. Create dataloader
    4. Train the model
    """
    # set seed for reproducibility
    set_seed(config.model.seed)
    # TODO: don't forget to remove
    # model_config = config.model.gpt
    model_config = config.model.gpt.config.small

    # Step 1: Load the data
    logger.info("Loading the data...")
    data_path = Path.cwd() / config.datasets.tiny_shakespeare.file_path
    with data_path.open() as fin:
        text = fin.read()
    logger.info("Data is loaded.")

    # Step 2: Prepare tokenizer and tokenize the data
    logger.info("Starting tokenizing...")
    tokenizer = CharTokenizer(corpus=text)
    data = torch.tensor(tokenizer.encode(text))
    logger.info("Tokenizing is done.")

    # Step 3: Create data loaders
    logger.info("Preparing data loaders...")
    train_data, test_data = train_test_split(data, 0.9)
    block_size = model_config.context_size
    batch_size = model_config.batch_size
    # train_dataset = Dataset(train_data, block_size)
    # test_dataset = Dataset(test_data, block_size)
    train_dataset = NextTokenRandomDataset(train_data, block_size, max_iter=5_000 * batch_size // 10)
    test_dataset = NextTokenRandomDataset(test_data, block_size, max_iter=550 * batch_size)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, num_workers=1)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, num_workers=1)
    logger.info("Data loaders are prepared.")

    # Step 4: Train the model
    logger.info("Staring training...")
    model = GPTLanguageModel(
        vocab_size=tokenizer.vocab_size,
        embeddings_size=model_config.embeddings_size,
        num_layers=model_config.num_layers,
        context_size=model_config.context_size,
        dropout=model_config.dropout,
        num_heads=model_config.num_heads,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=model_config.learning_rate)
    trainer = Trainer(model, optimizer, train_dataloader, test_dataloader, device=get_device())
    trainer.train(epochs=model_config.epochs)
    logger.info("Training is finished")


if __name__ == "__main__":
    train()
