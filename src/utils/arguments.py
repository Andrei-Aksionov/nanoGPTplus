import inspect
from collections.abc import Callable


def grab_arguments(func: Callable, kwargs: dict, ignore_kwargs: list | None = None) -> dict:
    """Return dictionary only with arguments that the func expects.

    Parameters
    ----------
    func : Callable
        function that expects some arguments;
        this helper function will grab only args that are expected
    kwargs : dict
        dictionary with keyword arguments
    ignore_kwargs : list | None, optional
        kwargs to ignore, by default None

    Returns
    -------
    dict
        kwargs that are expected by the provided function
    """
    ignore_kwargs = set(["self"] + (ignore_kwargs or []))
    expected_args = {kwarg for kwarg in inspect.getfullargspec(func).args if kwarg not in ignore_kwargs}

    return {k: v for k, v in kwargs.items() if k in expected_args}


class ArgumentSaverMixin:
    def save_arguments(self, ignore: list[str] = None) -> None:
        """Save all provided arguments into __dict__ by setattr.

        Parameters
        ----------
        ignore : list[str], optional
            list of arguments to ignore, by default None
        """
        ignore = set(["self"] + (ignore or []))
        # get local variables of the frame that called current one
        local_vars = inspect.currentframe().f_back.f_locals
        for arg_name, arg_value in local_vars.items():
            if arg_name not in ignore and not arg_name.startswith("_"):
                setattr(self, arg_name, arg_value)
