import subprocess

import pytest

from src import config
from src.data.downloader import download


@pytest.mark.smoke
@pytest.mark.order(1)
class TestDataset:
    def test_tiny_shakespeare_download(self) -> None:
        download(config.datasets.tiny_shakespeare)

    def test_tiny_shakespeare_download_script(self) -> None:
        completed_process = subprocess.run("python src/data/scripts/download_tiny_shakespeare.py".split())
        assert completed_process.returncode == 0
