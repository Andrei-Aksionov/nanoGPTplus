from typing import Literal

import torch


def get_device(prioritize_gpu: bool = True) -> Literal["cpu", "cuda"]:
    if not prioritize_gpu:
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"
