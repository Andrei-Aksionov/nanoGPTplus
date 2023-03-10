import subprocess

import pytest

from src import config


@pytest.mark.smoke
@pytest.mark.order(3)
class TestTraining:
    @pytest.mark.parametrize("model_size", list(config.model.bigram.size.keys()))
    def test_bigram_training(self, model_size: str) -> None:
        completed_process = subprocess.run(
            # fmt: off
            [
                "python", "src/model/train.py",
                "--model", "bigram",
                "--size", model_size,
                "--device", "cpu",
                "--dataset-fraction", "0.001",
            ],
            # fmt: on
        )
        assert completed_process.returncode == 0

    @pytest.mark.parametrize("model_size", list(config.model.gpt.size.keys()))
    def test_gpt_training(self, model_size: str) -> None:
        completed_process = subprocess.run(
            # fmt: off
            [
                "python", "src/model/train.py",
                "--model", "gpt",
                "--size", model_size,
                "--device", "cpu",
                "--dataset-fraction", "0.00001",
            ],
            # fmt: on
        )
        assert completed_process.returncode == 0
