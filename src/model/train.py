import argparse
import inspect
import multiprocessing
from pathlib import Path
from typing import Optional

import torch
from loguru import logger
from torch.utils.data import DataLoader

from src import config
from src.data import CharTokenizer, NextTokenDataset
from src.model import BigramLanguageModel, GPTLanguageModel, Trainer
from src.model.gpt_language_model.optimizers import CosineWarmupLRSchedular
from src.utils import get_device, grab_arguments, set_seed
from src.utils.arguments import RangeChecker
from src.utils.model import get_model_config, pickle_dump


def train(
    model_class: torch.nn.Module,
    device: Optional[str],
    size: str,
    dataset_fraction: Optional[float] = None,
) -> None:
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
    device: Optional[str]
        on what device to train, if not provided will try to figure out what device to use such as:
        if gpu (cuda or mps) is available will use it, if not - cpu
    size: str
        a model has two configs: small and large. Small is used for debug purpose as it's fast
    dataset_fraction: Optional[float]
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
    num_workers = min(multiprocessing.cpu_count(), config.dataloader.num_workers)
    train_dataloader = DataLoader(
        NextTokenDataset(train_data, model_config.context_size, dataset_fraction),
        batch_size=model_config.batch_size,
        num_workers=num_workers,
        shuffle=True,
    )
    test_dataloader = DataLoader(
        NextTokenDataset(test_data, model_config.context_size, dataset_fraction),
        batch_size=model_config.batch_size,
        num_workers=num_workers,
        shuffle=False,
    )
    logger.info("Data loaders are prepared.")

    # Step 4: Train the model
    # Step 4.1. Create model
    logger.info("Staring training...")
    model = model_class(vocab_size=tokenizer.vocab_size, **grab_arguments(model_class, model_config))
    # Step 4.2. Configure optimizer
    optimizer_parameters = model.optimizer_parameters if hasattr(model, "optimizer_parameters") else model.parameters()
    # new PyTorch nightly has a new 'fused' option for AdamW that is much faster
    if device == "cuda" and ("fused" in inspect.signature(torch.optim.AdamW).parameters):
        logger.debug("Using fused AdamW")
        extra_args = {"fused": True}
    else:
        extra_args = {}
    optimizer = torch.optim.AdamW(
        optimizer_parameters,
        lr=model_config.learning_rate,
        betas=model_config.betas,
        **extra_args,
    )
    # Step 4.3 Configure LR schedular
    if "warmup_iters" not in model_config or not model_config.warmup_iters:
        warmup_iters = int(len(train_dataloader) * 0.1)
    else:
        warmup_iters = model_config.warmup_iters
    logger.debug("Warmup iters: {}".format(warmup_iters))
    if "lr_decay_iters" not in model_config or not model_config.lr_decay_iters:
        lr_decay_iters = int(len(train_dataloader) * 0.95)
    else:
        lr_decay_iters = model_config.lr_decay_iters
    logger.debug("LR decay iters: {}".format(lr_decay_iters))
    lr_schedular = CosineWarmupLRSchedular(
        optimizer=optimizer,
        warmup_iters=warmup_iters,
        lr_decay_iters=lr_decay_iters,
    )
    # Step 4.4. Start training
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        train_dataloader=train_dataloader,
        eval_dataloader=test_dataloader,
        device=device or get_device(),
        lr_schedular=lr_schedular,
        grad_accumulation_steps=model_config.grad_accumulation_steps,
        checkpoint_model_path=model_config.checkpoint_model_path,
        tqdm_update_interval=model_config.tqdm_update_interval,
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
        choices=["small", "medium", "large"],
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
        choices=RangeChecker(0, 1, inclusive_start=False),
        help="For debugging purpose you can run training only on a fraction of the dataset",
        required=False,
        type=float,
    )
    args = vars(parser.parse_args())
    model_name = models[args.pop("model")]
    train(model_name, **args)


if __name__ == "__main__":
    main()
