from pathlib import Path

from omegaconf import OmegaConf

config = OmegaConf.load(Path(__file__).parents[1] / "config/config.yaml")
