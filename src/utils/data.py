import numpy as np


def train_test_split(
    data: list | np.ndarray,
    train_fraction: float,
) -> tuple[list, list] | tuple[np.ndarray, np.ndarray]:
    """Split array into train and test splits.

    Parameters
    ----------
    data : list | np.ndarray
        the whole data that needs to be splitted
    train_fraction : float
        what fraction of the data to use for training, the remaining
        will be used for testing

    Returns
    -------
    _type_
        _description_
    """
    test_split = int(len(data) * train_fraction)
    return data[:test_split], data[test_split:]
