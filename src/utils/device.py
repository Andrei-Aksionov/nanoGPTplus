import torch


def get_device(prioritize_gpu: bool = True) -> torch.device:
    """Return what device would be the best for the training.

    By default will try to return `gpu` and in case of failure -> rollback to `cpu`.

    Returns
    -------
    torch.device
        either `cuda` or `cpu`
    """
    if not prioritize_gpu:
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"
