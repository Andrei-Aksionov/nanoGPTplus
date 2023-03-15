from loguru import logger


def log_error(message: str, error_type: type = ValueError) -> None:
    """Apply helper function to log and raise an error with provided message.

    Parameters
    ----------
    message : str
        the message of the error
    error_type : type
        what error should be raise, by default ValueError

    """
    logger.error(message)
    raise error_type(message)
