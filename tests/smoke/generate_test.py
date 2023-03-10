import subprocess

import pytest

from src import config


@pytest.mark.smoke
@pytest.mark.order(4)
class TestTokenGeneration:
    @pytest.mark.parametrize("model_size", list(config.model.bigram.size.keys()))
    def test_bigram_token_generation(self, model_size: str) -> None:
        completed_process = subprocess.run(
            f"python src/model/generate.py --model bigram --size {model_size} "
            "--device cpu --max-new-tokens 10 --fix-seed".split(),
        )
        assert completed_process.returncode == 0

    @pytest.mark.parametrize("model_size", list(config.model.gpt.size.keys()))
    def test_gpt_token_generation(self, model_size: str) -> None:
        completed_process = subprocess.run(
            f"python src/model/generate.py --model gpt --size {model_size} "
            "--device cpu --max-new-tokens 10 --fix-seed".split(),
        )
        assert completed_process.returncode == 0
