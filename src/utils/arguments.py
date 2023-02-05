import inspect


def grab_arguments(func, args, ignore_args=None):
    ignore_args = set(["self"] + (ignore_args or []))
    expected_args = {arg for arg in inspect.getfullargspec(func).args if not arg in ignore_args}

    return {k: v for k, v in args.items() if k in expected_args}


class ArgumentSaverMixin:
    def save_arguments(self, ignore: list[str] = None):
        ignore = set(["self"] + (ignore or []))
        # get local variables of the frame that called current one
        local_vars = inspect.currentframe().f_back.f_locals
        for arg_name, arg_value in local_vars.items():
            if arg_name not in ignore and not arg_name.startswith("_"):
                setattr(self, arg_name, arg_value)
