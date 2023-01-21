from src import config
from src.data.downloader import download

if __name__ == "__main__":
    # Downloads tiny shakespeare dataset
    download(config.datasets.tiny_shakespeare)
