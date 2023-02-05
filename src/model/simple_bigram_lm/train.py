from pathlib import Path

import torch
from loguru import logger
from torch.utils.data import DataLoader

from src import config
from src.data.dataset import NextTokenDataset
from src.data.tokenizer import CharTokenizer
from src.model.simple_bigram_lm.bigram_lm import BigramLanguageModel
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
    # assign model's config to a variable
    model_config = config.model.bigram

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
    train_dataset = NextTokenDataset(train_data, block_size)
    test_dataset = NextTokenDataset(test_data, block_size)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, num_workers=1)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, num_workers=1)
    logger.debug("Data loaders are prepared.")

    # Step 4: Train the model
    logger.debug("Staring training...")
    model = BigramLanguageModel(vocab_size=tokenizer.vocab_size)
    optimizer = torch.optim.AdamW(model.parameters(), lr=model_config.learning_rate)
    trainer = Trainer(model, optimizer, train_dataloader, test_dataloader, device=get_device())
    trainer.train(epochs=model_config.epochs)
    logger.debug("Training is finished")


if __name__ == "__main__":
    train()
