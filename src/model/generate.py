import argparse

import torch
from loguru import logger

from src import config
from src.model import BigramLanguageModel, GPTLanguageModel
from src.utils import get_device, grab_arguments
from src.utils.model import get_model_config, load_checkpoint, pickle_load


def generate_new_tokens(model_class: torch.nn.Module, device: str | None, size: str, max_new_tokens: int) -> None:
    """Generate new tokens with help of pre-trained model.

    Parameters
    ----------
    model_class : torch.nn.Module
        which model to use in order to generate new tokens. The model should be pre-trained
        and weights should be stored in the folder that is specified in the config file
    device : str | None
        on which device to run token generation
    size : str
        the size of the model (small or large). Corresponding weights should exist in the folder
        that is specified in the config file
    max_new_tokens : int
        how many tokens to generate
    """
    # get device and model's config
    device = device or get_device()
    model_config = get_model_config(model_class, config, size)

    # load tokenizer and pre-trained models
    tokenizer = pickle_load(model_config.tokenizer_path)
    model = model_class(vocab_size=tokenizer.vocab_size, **grab_arguments(model_class, model_config))
    load_checkpoint(model_config.checkpoint_model_path, model)
    model.to(device)

    # generate tokens
    logger.debug("Generating tokens on '{}' device".format(device))
    tokens = tokenizer.encode(" ")
    context = torch.tensor(tokens, device=device).unsqueeze(dim=0)
    new_tokens = tokenizer.decode(model.generate(context, max_new_tokens=max_new_tokens).squeeze().tolist())
    logger.info("New generated tokens: {}".format(new_tokens))


def main() -> None:
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
        "--max-new-tokens",
        default=100,
        help="How many new tokens do you want to generate",
        required=False,
        type=int,
    )
    args = parser.parse_args()
    generate_new_tokens(models[args.model], args.device, args.size, args.max_new_tokens)


if __name__ == "__main__":
    main()
