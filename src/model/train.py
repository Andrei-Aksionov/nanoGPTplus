import argparse
from pathlib import Path

import torch
from loguru import logger
from torch.utils.data import DataLoader

from src import config
from src.data import CharTokenizer, NextTokenDataset, NextTokenRandomDataset
from src.model import BigramLanguageModel, GPTLanguageModel, Trainer
from src.utils import get_device, set_seed
from src.utils.arguments import grab_arguments


def train(model_class: torch.nn.Module, is_debug: bool) -> None:
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

    model_class_name = model_class.__name__
    if model_class_name == "BigramLanguageModel":
        model_config = config.model.bigram
    elif model_class_name == "GPTLanguageModel":
        model_config = config.model.gpt.size.small if is_debug else config.model.gpt.size.large
    else:
        msg = f"There is no config for class '{model_class_name}'"
        logger.critical(msg)
        raise ValueError(msg)

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
    test_split = int(len(data) * config.model.test_split)
    train_data, test_data = data[:test_split], data[test_split:]
    block_size = model_config.context_size
    batch_size = model_config.batch_size
    # train_dataset = NextTokenDataset(train_data, block_size)
    # test_dataset = NextTokenDataset(test_data, block_size)
    train_dataset = NextTokenRandomDataset(train_data, block_size, max_iter=5_000 * batch_size // 100)
    test_dataset = NextTokenRandomDataset(test_data, block_size, max_iter=550 * batch_size // 10)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, num_workers=1)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, num_workers=1)
    logger.info("Data loaders are prepared.")

    # Step 4: Train the model
    logger.info("Staring training...")
    model = model_class(vocab_size=tokenizer.vocab_size, **grab_arguments(model_class, model_config))
    optimizer = torch.optim.AdamW(model.parameters(), lr=model_config.learning_rate)
    trainer = Trainer(model, optimizer, train_dataloader, test_dataloader, device=get_device())
    trainer.train(epochs=model_config.epochs)
    logger.info("Training is finished")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train bigram or GPT language model")
    models = {
        "bigram": BigramLanguageModel,
        "gpt": GPTLanguageModel,
    }
    parser.add_argument("--model", "-m", type=str, help="Bigram or GPT", choices=list(models), required=True)
    parser.add_argument("--debug", "-d", action="store_true", help="Do you want run fast train for debug purpose?")
    args = parser.parse_args()
    train(models[args.model], args.debug)


if __name__ == "__main__":
    main()
