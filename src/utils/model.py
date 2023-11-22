import pickle
from pathlib import Path

import torch

from src.utils import log_error


def pickle_dump(file_to_save: object, path: str) -> None:
    """Pickle any file into provided filepath.

    Parameters
    ----------
    file_to_save : object
        a file to pickle
    path : str
        where to pickle the file
    """
    path = Path(__file__).parents[2] / path
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as fout:
        pickle.dump(file_to_save, fout)


def pickle_load(path: str) -> object:
    """Load pickled file from provided filepath.

    Parameters
    ----------
    path : str
        where pickled file is stored

    Returns
    -------
    object
        loaded file
    """
    path = Path(__file__).parents[2] / path
    with path.open("rb") as fin:
        return pickle.load(fin)


def save_checkpoint(state: dict, path: str) -> None:
    """Save current state of the model with parameters.

    Parameters
    ----------
    state : dict
        parameters of the model represented as a dict
    path: str
        where and in which file to save the model's state
    """
    path = Path(__file__).parents[2] / path
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)


def load_checkpoint(path: str, model: torch.nn.Module) -> None:
    """Load saved state of the model with parameters.

    Parameters
    ----------
    path: str
        where model's weights are stored
    model : torch.nn.Module
        model in which saved parameter values will be loaded
    """
    path = Path(__file__).parents[2] / path
    model.load_state_dict(torch.load(path))


def get_model_config(model_class: torch.nn.Module, config: dict, size: str) -> dict:
    """Return dictionary with model's parameters.

    Parameters are sourced from the config file and selected by model class name and it's size.

    Parameters
    ----------
    model_class : torch.nn.Module
        class of the model
    config : dict
        main config file with parameters of all models
    size : str
        size of the model (small or large)

    Returns
    -------
    dict
        model's parameters

    Raises
    ------
    ValueError
        if there is no config in the config file for the provided model class
    """
    model_class_name = model_class.__name__
    model_config = {"BigramLanguageModel": config.model.bigram, "GPTLanguageModel": config.model.gpt}.get(
        model_class_name
    )
    if model_config is None:
        log_error(f"There is no config for class '{model_class_name}'")
    return model_config.size[size]
