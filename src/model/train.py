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
from src.model import BigramLanguageModel, CosineWarmupLRScheduler, GPTLanguageModel, Trainer
from src.model.gpt_language_model.peft.lora import lora, mark_only_lora_as_trainable
from src.utils import RangeChecker, get_device, get_model_config, grab_arguments, pickle_dump, set_seed


# TODO: perhaps it's about time to split this func into a set of smaller ones
def train(  # noqa: PLR0915
    model_class: torch.nn.Module,
    device: Optional[str],
    size: str,
    dataset_fraction: Optional[float] = None,
    use_lora: bool = False,
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
    # set up logger to write also in a file
    logger.add(config.logs.training, **config.logs.logger_kwargs)

    # set seed for reproducibility
    set_seed(config.seed)
    logger.debug("Random seed is fixed for training.")

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
    # NOTE: this is just an example of how to use LoRA with the model.
    # LoRA should be used with pretrained weights and right now only training from scratch is supported.
    # That's why by default it's disabled in the config file.
    if model_class == GPTLanguageModel and (model_config.use_lora or use_lora):
        with lora(
            r=model_config.lora_rank,
            alpha=model_config.lora_alpha,
            dropout=model_config.lora_dropout,
            enabled=model_config.use_lora or use_lora,
        ):
            model = model_class(vocab_size=tokenizer.vocab_size, **grab_arguments(model_class, model_config))
        mark_only_lora_as_trainable(model)
    else:
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
        optimizer_parameters, lr=model_config.learning_rate, betas=model_config.betas, **extra_args
    )

    # Step 4.3 Configure LR schedular
    # if warmup/lr_decay iters is None - set default
    # if it's a float - use it as a portion
    # else - use as is
    warmup_iters = model_config.get("warmup_iters")
    if warmup_iters is None:
        warmup_iters = int(len(train_dataloader) * 0.1)
    elif isinstance(warmup_iters, float):
        warmup_iters = int(len(train_dataloader) * warmup_iters)
    logger.debug("LR warmup iters: {}".format(warmup_iters))
    lr_decay_iters = model_config.get("lr_decay_iters")
    if lr_decay_iters is None:
        lr_decay_iters = int(len(train_dataloader) * 0.95)
    elif isinstance(lr_decay_iters, float):
        lr_decay_iters = int(len(train_dataloader) * lr_decay_iters)
    logger.debug("LR decay iters: {}".format(lr_decay_iters))
    lr_scheduler = CosineWarmupLRScheduler(
        optimizer=optimizer, warmup_iters=warmup_iters, lr_decay_iters=lr_decay_iters
    )
    # Step 4.4. Start training
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        train_dataloader=train_dataloader,
        eval_dataloader=test_dataloader,
        device=device or get_device(),
        lr_scheduler=lr_scheduler,
        grad_accumulation_steps=model_config.grad_accumulation_steps,
        checkpoint_model_path=model_config.checkpoint_model_path,
        tqdm_update_interval=model_config.tqdm_update_interval,
    )
    trainer.train(epochs=model_config.epochs)
    logger.info("Training is finished")


def main() -> None:
    """Train either GPT or a simple bigram language model on tiny-shakespeare dataset."""
    # main parser will store subparsers, shared parser - arguments that are shared between subparsers
    main_parser = argparse.ArgumentParser(
        description="Train bigram or GPT language model", formatter_class=argparse.RawTextHelpFormatter
    )
    shared_parser = argparse.ArgumentParser(add_help=False)
    # ordering matters: first shared arguments, then - subparsers
    # ---------- shared arguments ----------
    shared_parser.add_argument(
        "--device", help="Optionally you can select device on which the model will be trained", required=False, type=str
    )
    shared_parser.add_argument(
        "--dataset-fraction",
        choices=RangeChecker(0, 1, inclusive_start=False),
        help="For debugging purpose you can run training only on a fraction of the dataset",
        required=False,
        type=float,
    )
    # ---------- subparsers ----------
    subparsers = main_parser.add_subparsers(dest="model")
    # bigram subparser
    bigram_subparser = subparsers.add_parser("bigram", parents=[shared_parser])
    bigram_subparser.add_argument(
        "--size", "-s", choices=["large"], help="The size of the Bigram model", required=True, type=str
    )
    # gpt subparser
    gpt_subparser = subparsers.add_parser("gpt", parents=[shared_parser])
    gpt_subparser.add_argument(
        "--size", "-s", choices=["small", "medium", "large"], help="The size of the GPT model", required=True, type=str
    )
    gpt_subparser.add_argument(
        "--use-lora",
        help="Forces to use LoRA no matter what is set in the config file.",
        action="store_true",
        required=False,
    )

    # combining 'help' output from both argparsers
    shared_parser_help = (
        shared_parser.format_help().replace("optional arguments:", "").replace(shared_parser.format_usage(), "")
    )
    shared_parser_help = f"{' Arguments common to all sub-parsers '.center(100, '-')}{shared_parser_help}"
    main_parser.epilog = shared_parser_help

    # parser arguments
    args = vars(main_parser.parse_args())
    model_name = {"bigram": BigramLanguageModel, "gpt": GPTLanguageModel}[args.pop("model")]

    # run model training
    train(model_name, **args)


if __name__ == "__main__":
    main()
