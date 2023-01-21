from pathlib import Path

import requests
from loguru import logger
from omegaconf import DictConfig


def download(config: DictConfig) -> Path:
    """Download file into specified folder.

    Parameters
    ----------
    config : DictConfig
        omegaconf's dictionary with two keys: url and folder
        url is from where to download the file
        folder - in which folder to put the downloaded file

    Returns
    -------
    Path
        where the downloaded file is
    """
    url = config.url
    dst_folder = Path.cwd() / config.folder
    dst_folder.mkdir(parents=True, exist_ok=True)

    logger.debug("Downloading {} into {}".format(url, dst_folder))
    response = requests.get(url)
    with open(dst_folder / Path(url).name, "wb") as fout:
        fout.write(response.content)
    logger.debug("Downloading is finished")
