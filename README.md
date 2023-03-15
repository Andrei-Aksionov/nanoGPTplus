
<p>
    <h2 align="center">Welcome to NanoGPT+ in PyTorch</h2>
    <h5 align="center">Knock-off edition (but with enchantments)<h5>
</p>

[![Python 3.8](https://img.shields.io/badge/python-3.8-blue.svg)](https://www.python.org/downloads/release/python-380/)
[![Python 3.9](https://img.shields.io/badge/python-3.9-blue.svg)](https://www.python.org/downloads/release/python-390/)
[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/release/python-310/)
[![test](https://github.com/Andrei-Aksionov/nanoGPTplus/actions/workflows/test.yaml/badge.svg)](https://github.com/Andrei-Aksionov/nanoGPTplus/actions/workflows/test.yaml)

***

In this repository I want to rewrite the code for `nanoGPT` presented by Andrej Karpathy in [this video](https://www.youtube.com/watch?v=kCc8FmEb1nY). The original code is in a state that is suitable for rapid prototyping, while the code in this repository in my opinion is more mature (with docstrings, comments of what is exactly going on, readme for architectures, ...) hence the name - nanoGPT+ (you can read it as a very-very small plus :laughing:)

The purpose of it is to better understand how Transformer architecture works by actually writing code and, if possible, making it better (or at least to make it work with as few issues as possible).

> **Note**: while the code in this repository reflects almost all the logic of the original one, because of lack of access to GPU (or moreover to a multiple GPUs/nodes with multiple GPUs) I haven't added GPU specific code, so if you have one (GPU or even a node) then you should look at the [original repo](https://github.com/karpathy/nanoGPT).

<p align=center><img src="references/readme/amazon_prime.jpg"></p>

# Project structure

- **data**
  - **raw**: there will be stored unprocessed data
- **notebooks**: notebooks are also can be rendered by [nbviewer](https://nbviewer.org/)
  - **EDA**: exploratory data analysis notebooks where one can gain insights into the dataset
    - *tiny_shakespeare.ipynb*: EDA of the tiny-shakespeare dataset
  - **examples**:
    - *bigram_model_training.ipynb*: example of how this language model can be trained
    - *gpt_model_training.ipynb*: example of how to train gpt language model
- **src**
  - **config**
    - *config.yaml*: config file
  - **data**
    - **scripts**:
      - *download_tiny_shakespeare*: basically one-liner to download the dataset
    - *dataset.py*: contains custom nn.Dataset
    - *downloader.py*: downloads the data from the web
    - *tokenizer.py*: transform text into integers by corresponding mappings in the vocabulary
  - **model**
    - **bigram_language_mode**
      - *bigram_lm.py*: implementation of simple bigram language model
      - **README.md*: notes about architecture of bigram language model
    - **gpt_language_model**
      - *attention.py*: single head, multi-head and casual self-attentions
      - *feed_forward.py*: feed-forward layer (position-wise MLP) of transformer block
      - *gpt.py*: the whole GPT architecture
      - *optimizers.py*: contain custom optimizers and learning rate schedulers
      - *README.md*: notes about GPT as a whole and attention in particular
      - *transformer_block.py*: building block of GPT including self-attention and feed-forward
    - *generate.py*: code for generating new tokens with help of pre-trained model
    - *train.py*: code to train language model
    - *trainer.py*: code to do all the necessary step for training and evaluating the model
  - **utils**: various utils files
- **tests**
  - **smoke**: include smoke tests for quick testing of repo functionality (download data, train and generate new tokens)
- *pyproject.toml*: package dependencies are stored here and managed py [Poetry](https://python-poetry.org/)

# How to use it

1. Install all the required packages. In this project all the python dependencies are managed by [Poetry](https://python-poetry.org/) and are stored in "pyproject.toml" file (in this file also specified required version of python). After `poetry` is installed and virtual environment is created (in case you don't want poetry to create it [automatically](https://python-poetry.org/docs/configuration/#virtualenvscreate)), run:

    ```bash
    poetry install
    ```

2. This project uses the same tiny-shakespeare dataset as in Andrej Karpathy's version. In order to download the data from the web just run:

    ```python
    python src/data/scripts/download_tiny_shakespeare.py
    ```

    ... or download your favorite dataset, the only thing it has to be a plain text (.txt format). In addition to that change the path to the file in `src/config/config.yaml` or add sub-section in the `dataset` section with new dataset name and do not forget change it in the training scripts.

3. Rerun EDA notebook [optional]

    If you work with you custom dataset and it is in plain text format, simply rerun or copy and rerun `notebooks/EDA/tiny-shakespeare.ipynb` notebook to gain insight into the data.

4. Run training via script:

    Train script accepts multiple arguments:
    - `[bigram, gpt]`: choose model type.
    - `--size [small, medium, large]` (small is good for debugging alongside with dataset fraction). **Note**: for bigram model only large size is available
    - `--device [cpu, cuda, mps]` **[Optional]**: if not provided will try to detect automatically (GPU first and if it's not available - fallback to cpu).
    - `--dataset-fraction` **[Optional]**: useful if one wants to quickly run training for debugging (affects both training and testing datasets). If not provided the whole dataset will be used.

    **Arguments 1 and 2 are required.**

    ```bash
    python src/model/train.py gpt --size large
    ```

    ... or in one of the example notebooks in `notebooks/examples/*_model_training.ipynb`.

5. Run new token generation:

    Generation script accepts multiple arguments:
    - `[bigram, gpt]`: choose model type .
    - `--size [small, medium, large]` (small is good for debugging). **Note**: for bigram model only large size is available
    - `--gpt2-config [gpt2, gpt2-medium,  gpt2-large, gpt2-xl]`: load weights from pretrained model
    - `--device [cpu, cuda, mps]` **[Optional]**: if not provided will try to detect automatically (GPU first and if it's not available - fallback to cpu).
    - `--max-new-tokens` **[Optional]**: number of tokens to generate. If not provided the default value will be used, which is 100.
    - `--temperature` **[Optional]**: if the value is above 1 - less randomness in token generation, if less then 1 - more randomness ([explained here](https://ai.stackexchange.com/questions/32477/what-is-the-temperature-in-the-gpt-models)). Usefull if you want to make token prediction more diverse, but the cost of it will be lower accuracy of generated tokens.
    - `--fix-seed` **[Optional]**: if provided will fix to the seed specified in config file
    - `--continue-words` **[Optional]**: if provided the model will generate new tokens that continue provided ones

    ```bash
    python src/model/generate.py gpt --size large --max-new-tokens 100
    ```

    Also possible to load GPT2 pretrained weights into a model and generate new tokens. In order to do that instead of providing size of the model provide `--gpt2-config` argument:

    ```bash
    python src/model/generate.py gpt --gpt2-config gpt2 --max-new-tokens --temperature 0.8 --continue-tokens "Hello world!"

## Run tests

[Pytest](https://github.com/pytest-dev/pytest) framework is used for tests execution so in order to run all tests simply type:

```bash
pytest
```

> **Note**: the command above will run all the tests.

There are two types of tests: fast (smoke) and slow ones. Slow tests will load large size of GPT2 models. If one wants to run only fast test:

```bash
pytest -m smoke
```

If one wants to see also standard output in pytest logs (for example for tests of training):

```bash
pytest --include=sys
```

***

## Additional: pre-commit hooks

In order to install pre-commit hooks run:

```bash
pre-commit install
```

Pre-commit hooks will be executed before each commit. In addition all the pre-commit hooks will be run per each PR via github-workflow (no need to add or change anything).

The list of all hooks one can find in a config fils: `.pre-commit-config.yaml`

**Note**: for the sake of speed pre-commit hooks will be executed only on changed files. If it's needed to run on all files execute:

```bash
pre-commit run --all-files
```
