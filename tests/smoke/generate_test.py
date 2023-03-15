import subprocess

import pytest

from src import config


@pytest.mark.order(4)
class TestTokenGeneration:
    @pytest.mark.smoke
    @pytest.mark.parametrize("model_size", list(config.model.bigram.size.keys()))
    def test_bigram_token_generation(self, model_size: str) -> None:
        completed_process = subprocess.run(
            # fmt: off
            [
                "python", "src/model/generate.py", "bigram",
                "--size", model_size,
                "--device", "cpu",
                "--max-new-tokens", "10",
                "--continue-tokens", "Hello world ",
                "--temperature", "1.0",
                "--fix-seed",
            ],
            # fmt: on
        )
        assert completed_process.returncode == 0

    @pytest.mark.smoke
    @pytest.mark.parametrize("model_size", list(config.model.gpt.size.keys()))
    def test_gpt_token_generation(self, model_size: str) -> None:
        completed_process = subprocess.run(
            # fmt: off
            [
                "python", "src/model/generate.py", "gpt",
                "--size", model_size,
                "--device", "cpu",
                "--max-new-tokens", "10",
                "--continue-tokens", "Hello world ",
                "--temperature", "0.8",
                "--fix-seed",
            ],
            # fmt: on
        )
        assert completed_process.returncode == 0

    @pytest.mark.smoke
    @pytest.mark.parametrize("model_type", ["gpt2"])
    def test_gpt2_pretrained_token_generation_fast(self, model_type: str) -> None:
        completed_process = subprocess.run(
            # fmt: off
            [
                "python", "src/model/generate.py", "gpt",
                "--gpt2-config", model_type,
                "--device", "cpu",
                "--max-new-tokens", "10",
                "--continue-tokens", "Hello world ",
                "--temperature", "0.8",
                "--fix-seed",
            ],
            # fmt: on
        )
        assert completed_process.returncode == 0

    @pytest.mark.slow
    @pytest.mark.parametrize("model_type", ["gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"])
    def test_gpt2_pretrained_token_generation_slow(self, model_type: str) -> None:
        completed_process = subprocess.run(
            # fmt: off
            [
                "python", "src/model/generate.py", "gpt",
                "--gpt2-config", model_type,
                "--device", "cpu",
                "--max-new-tokens", "10",
                "--continue-tokens", "Hello world ",
                "--temperature", "0.8",
                "--fix-seed",
            ],
            # fmt: on
        )
        assert completed_process.returncode == 0
