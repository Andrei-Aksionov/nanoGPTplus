import argparse
from pathlib import Path

import torch
from loguru import logger
from torch.utils.data import DataLoader

from src import config
from src.data import CharTokenizer, NextTokenDataset
from src.model import BigramLanguageModel, GPTLanguageModel, Trainer
from src.utils import get_device, grab_arguments, set_seed
from src.utils.model import get_model_config, pickle_dump


def train(model_class: torch.nn.Module, device: str | None, size: str, dataset_fraction: float | None = None) -> None:
    """Train a language model.

    Performs 4 steps:
    1. Load the data
    2. Create a tokenizer
    3. Create a dataloader
    4. Train the model

    Parameters
    ----------
    model_class : torch.nn.Module
        what language model to use
    device: str | None
        on what device to train, if not provided will try to figure out what device to use such as:
        if gpu (cuda or mps) is available will use it, if not - cpu
    size: str
        a model has two configs: small and large. Small is used for debug purpose as it's fast
    dataset_fraction: float | None
        for debugging purposes one might want to run training only on a small fraction of a dataset
    """
    # set seed for reproducibility
    set_seed(config.seed)

    # assign model's config to a variable
    model_config = get_model_config(model_class, config, size)

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

    # Step 2.1. Save tokenizer
    logger.info("Saving tokenizer...")
    pickle_dump(tokenizer, model_config.tokenizer_path)
    logger.info("Tokenizer is saved.")

    # Step 3: Create data loaders
    logger.info("Preparing data loaders...")
    # Step 3.1. Split data into train/test split
    test_split = int(len(data) * config.dataloader.test_split)
    train_data, test_data = data[:test_split], data[test_split:]
    # Step 3.2. Create data loaders
    train_dataloader = DataLoader(
        NextTokenDataset(train_data, model_config.context_size, dataset_fraction),
        batch_size=model_config.batch_size,
        num_workers=config.dataloader.num_workers,
    )
    test_dataloader = DataLoader(
        NextTokenDataset(test_data, model_config.context_size, dataset_fraction),
        batch_size=model_config.batch_size,
        num_workers=config.dataloader.num_workers,
    )
    logger.info("Data loaders are prepared.")

    # Step 4: Train the model
    logger.info("Staring training...")
    model = model_class(vocab_size=tokenizer.vocab_size, **grab_arguments(model_class, model_config))
    optimizer = torch.optim.AdamW(model.parameters(), lr=model_config.learning_rate)
    device = device if device else get_device()
    trainer = Trainer(
        model,
        optimizer,
        train_dataloader,
        test_dataloader,
        device=device,
        checkpoint_model_path=model_config.checkpoint_model_path,
    )
    trainer.train(epochs=model_config.epochs)
    logger.info("Training is finished")


def main() -> None:
    """Train either GPT or a simple bigram language model on tiny-shakespeare dataset."""
    parser = argparse.ArgumentParser(description="Train bigram or GPT language model")
    models = {"bigram": BigramLanguageModel, "gpt": GPTLanguageModel}
    parser.add_argument(
        "--model",
        "-m",
        choices=list(models),
        help="Bigram or GPT",
        required=True,
        type=str,
    )
    parser.add_argument(
        "--size",
        "-s",
        choices=["small", "large"],
        help="The size of the model (small or large)",
    )
    parser.add_argument(
        "--device",
        help="Optionally you can select device on which the model will be trained",
        required=False,
        type=str,
    )
    parser.add_argument(
        "--dataset-fraction",
        help="For debugging purpose you can run training only on a fraction of the dataset",
        required=False,
        type=float,
    )
    args = parser.parse_args()
    train(models[args.model], args.device, args.size, args.dataset_fraction)


if __name__ == "__main__":
    main()
