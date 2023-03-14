import argparse
from time import perf_counter
from typing import Optional

import tiktoken
import torch
from loguru import logger

from src import config
from src.model import BigramLanguageModel, GPTLanguageModel
from src.utils import (
    RangeChecker,
    get_device,
    get_model_config,
    grab_arguments,
    load_checkpoint,
    log_error,
    pickle_load,
    set_seed,
)


def generate_new_tokens(
    model_class: torch.nn.Module,
    device: Optional[str],
    size: Optional[str],
    gpt2_config: Optional[str],
    max_new_tokens: int,
    temperature: float,
    fix_seed: bool,
    continue_tokens: str,
) -> None:
    """Generate new tokens with help of pre-trained model.

    Parameters
    ----------
    model_class : torch.nn.Module
        which model to use in order to generate new tokens. The model should be pre-trained
        and weights should be stored in the folder that is specified in the config file
    device : Optional[str]
        on which device to run token generation
    size : Optional[str]
        the size of the model (small or large). Corresponding weights should exist in the folder
        that is specified in the config file
    gpt2_config: Optional[str]
        weights from this config will be loaded from huggingface.
    max_new_tokens : int
        how many tokens to generate
    temperature: float
        temperature >= 1.0 - smaller randomness (small variations), temperature < 1.0 - higher randomness
    fix_seed: bool
        might be useful for debugging to have the same output every time, if so, then set fix_seed=True
    continue_tokens: str
        new token should be generated as a continuation of tokens in 'continue_tokens' variable
    """
    # set up logger to write also in a file
    logger.add(config.logs.generation, **config.logs.logger_kwargs)

    # if specified set seed for reproducibility
    if fix_seed:
        set_seed(config.seed)
    logger.debug(f"Random seed is {'' if fix_seed else 'NOT '}fixed for token generation.")

    # get device and model's config
    device = device or get_device()

    # Bigram should have size, GPT - either size of gpt2_config, not both
    if model_class == BigramLanguageModel and size is None:
        log_error("For Bigram language model size has to be provided")
    if model_class == GPTLanguageModel:
        if not size and not gpt2_config:
            log_error("For GPT language model either size or gpt2_config has to be provided, but both are empty.")
        elif size and gpt2_config:
            log_error(
                "For GPT language model either size or gpt2_config has to be provided, not both, "
                f"but was provided size={size} and gpt2_config={gpt2_config}"
            )

    # if all checks are passed, that means that either size of gpt2_config is provided
    if size is not None:
        model_config = get_model_config(model_class, config, size)
        # load tokenizer and pre-trained models
        tokenizer = pickle_load(model_config.tokenizer_path)
        model = model_class(vocab_size=tokenizer.vocab_size, **grab_arguments(model_class, model_config))
        load_checkpoint(model_config.checkpoint_model_path, model)
    else:
        tokenizer = tiktoken.get_encoding("gpt2")
        model = GPTLanguageModel.from_pretrained(gpt2_config)
    model.to(device)
    model.eval()

    # generate tokens
    start_time = perf_counter()
    logger.debug("Generating tokens on '{}' device".format(device))
    tokens = tokenizer.encode(continue_tokens)
    context = torch.tensor(tokens, device=device)[None, ...]
    new_tokens = tokenizer.decode(
        model.generate(
            context,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )
        .squeeze()
        .tolist(),
    )
    logger.info("New generated tokens: {}".format(new_tokens))
    logger.debug("Token generation took: {:.4f} seconds".format(perf_counter() - start_time))


def main() -> None:
    # flake8: noqa
    """Generate new tokens from either GPT or a simple bigram language model."""
    parser = argparse.ArgumentParser(description="Generate new tokens")
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
        required=False,
        type=str,
    )
    parser.add_argument(
        "--gpt2-config",
        choices=["gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"],
        help="GPT2 config with pretrained weights",
        required=False,
        type=str,
    )
    parser.add_argument(
        "--device",
        help="Optionally you can select device on which the model will be trained",
        required=False,
        type=str,
    )
    parser.add_argument(
        "--max-new-tokens",
        default=100,
        help="How many new tokens do you want to generate",
        required=False,
        type=int,
    )
    parser.add_argument(
        "--temperature",
        default=1.0,
        choices=RangeChecker(0, float("inf"), inclusive_start=False),
        help="Temperature >= 1.0 - smaller randomness (small variations), temperature < 1.0 - higher randomness",
        required=False,
        type=float,
    )
    parser.add_argument(
        "--fix-seed",
        help="Make token generation deterministic",
        action="store_true",
        required=False,
    )
    parser.add_argument(
        "--continue-tokens",
        default=" ",
        help="Generation should continue these tokens",
        required=False,
        type=str,
    )
    args = vars(parser.parse_args())
    model_name = models[args.pop("model")]
    generate_new_tokens(model_name, **args)


if __name__ == "__main__":
    main()
