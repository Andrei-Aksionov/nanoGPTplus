from pathlib import Path

import torch
from loguru import logger
from torch.utils.data import DataLoader

from src import config
from src.data.dataset import Dataset
from src.data.tokenizer import CharTokenizer
from src.model.gpt_lm.gpt import GPT
from src.model.trainer import Trainer
from src.utils.data import train_test_split
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
    model_config = config.model.gpt.debug

    # Step 1: Load the data
    logger.debug("Loading the data...")
    data_path = Path.cwd() / config.datasets.tiny_shakespeare.file_path
    with data_path.open() as fin:
        text = fin.read()
    logger.debug("Data is loaded.")

    # Step 2: Prepare tokenizer and tokenize the data
    logger.debug("Starting tokenizing...")
    tokenizer = CharTokenizer(corpus=text)
    data = torch.tensor(tokenizer.encode(text))
    logger.debug("Tokenizing is done.")

    # Step 3: Create data loaders
    logger.debug("Preparing data loaders...")
    train_data, test_data = train_test_split(data, 0.9)
    block_size = model_config.block_size
    batch_size = model_config.batch_size
    train_dataset = Dataset(train_data, block_size)
    test_dataset = Dataset(test_data, block_size)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, num_workers=1)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, num_workers=1)
    logger.debug("Data loaders are prepared.")

    # Step 4: Train the model
    logger.debug("Staring training...")
    model = GPT(
        vocab_size=tokenizer.vocab_size,
        n_embed=model_config.n_embed,
        n_layers=model_config.n_layers,
        block_size=model_config.block_size,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=model_config.learning_rate)
    trainer = Trainer(model, optimizer, train_dataloader, test_dataloader)
    trainer.train(epochs=model_config.epochs)
    logger.debug("Training is finished")


if __name__ == "__main__":
    train()
