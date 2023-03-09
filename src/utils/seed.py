import os
import random

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """For reproducibility tries to fix every possible seed for random generators.

    Parameters
    ----------
    seed : int
        some arbitrary number to fix the seed
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)  # noqa: NPY002
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
