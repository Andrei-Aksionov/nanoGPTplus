import torch


def get_device(*, prioritize_gpu: bool = True) -> str:
    """Return what device would be the best for the training.

    By default will try to return `gpu` and in case of failure -> rollback to `cpu`.

    Parameters
    ----------
    prioritize_gpu : bool, optional
        should try use gpu first, by default True

    Returns
    -------
    str
        either `cuda`, `mps` or `cpu`
    """
    if prioritize_gpu:
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
    return "cpu"
