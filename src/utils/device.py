import torch


def get_device(prioritize_gpu: bool = True):
    if not prioritize_gpu:
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"
