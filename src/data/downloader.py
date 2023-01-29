from pathlib import Path

import requests
from loguru import logger
from omegaconf import DictConfig


def download(config: DictConfig, override_if_exists: bool = True) -> Path:
    """Download file into specified folder.

    Parameters
    ----------
    config : DictConfig
        omegaconf's dictionary with three keys: url, folder and filename
        url is from where to download the file
        folder - in which folder to put the downloaded file
    override_if_exists: bool
        if True will download even if file with such name already exists

    Returns
    -------
    Path
        where the downloaded file is
    """
    url = config.url
    dst_folder = Path.cwd() / config.folder
    dst_folder.mkdir(parents=True, exist_ok=True)

    logger.debug("Downloading {} into {}".format(url, dst_folder))
    file_path = dst_folder / config.filename
    if file_path.exists() and not override_if_exists:
        logger.debug("File already exists and specified to not override: not downloading")
        return file_path
    response = requests.get(url)
    with open(file_path, "wb") as fout:
        fout.write(response.content)
    logger.debug("Downloading is finished")
