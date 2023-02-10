import pickle
from pathlib import Path
from typing import Any

import torch


def pickle_dump(file_to_save: Any, path: str) -> None:
    path = Path(__file__).parents[2] / path
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as fout:
        pickle.dump(file_to_save, fout)


def pickle_load(path: str) -> Any:
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
    model : torch.nn.Module
        model in which saved parameter values will be loaded
    """
    # model.load_state_dict(checkpoint["state_dict"])
    path = Path(__file__).parents[2] / path
    model.load_state_dict(torch.load(path))
