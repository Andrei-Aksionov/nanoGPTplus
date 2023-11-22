import subprocess

import pytest

from src import config


@pytest.mark.smoke()
@pytest.mark.order(3)
class TestTraining:
    @pytest.mark.parametrize("model_size", list(config.model.bigram.size.keys()))
    def test_bigram_training(self, model_size: str) -> None:
        completed_process = subprocess.run(
            # fmt: off
            [
                "python", "src/model/train.py", "bigram",
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
                "python", "src/model/train.py", "gpt",
                "--size", model_size,
                "--device", "cpu",
                "--dataset-fraction", "0.00001",
            ],
            # fmt: on
        )
        assert completed_process.returncode == 0


# LoRA testing should be done after training without it to not confuse saved checkpoints
@pytest.mark.smoke()
@pytest.mark.order(5)
# Smoke tests of Low Ranking Adaptation (LoRA)
class TestTrainingWithLoRA:
    @pytest.mark.parametrize("model_size", list(config.model.gpt.size.keys()))
    def test_gpt_small_training_with_lora(self, model_size: str) -> None:
        completed_process = subprocess.run(
            # fmt: off
            [
                "python", "src/model/train.py", "gpt",
                "--size", model_size,
                "--device", "cpu",
                "--dataset-fraction", "0.00001",
                "--use-lora",
            ],
            # fmt: on
        )
        assert completed_process.returncode == 0
