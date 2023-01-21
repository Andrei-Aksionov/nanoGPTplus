In this repository I want to rewrite the code for 'nanoGPT' presented by Andrej Karpathy.
It will be done because I believe if you want to properly understand the code - rewrite it and, if possible, make it better.


## Install

As this project uses pyproject.toml - [poetry](https://python-poetry.org/docs/) has to be installed.

Also take a look at the required python version (described in **pyproject.toml** file).

In order to install all required packages run this command (when you are in the folder with pyproject.toml file).

```sh
poetry install
```


***

## Additional: git pre-commit hook

In order to run `black` formatter and `flake8` linter before each commit you need to add them into `.git/hooks` folder either manually or with helper script:

```bash
sh .add_git_hooks.sh`
```

This script will put `pre-commit` file into `.git/hooks` folder of the project and make it executable.
