<p>
    <h2 align="center">Welcome to NanoGPT in PyTorch</h2>
    <h5 align="center">Knock-off edition<h5>
</p>

***

In this repository I want to rewrite the code for `nanoGPT` presented by Andrej Karpathy in [this video](https://www.youtube.com/watch?v=kCc8FmEb1nY).

The purpose of it is to better understand how Transformer architecture works by actually writing code and, if possible, making it better (or at least to make it work with as few issues as possible).

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
      - *attention.py*: single head and multi-head self-attention
      - *feed_forward.py*: feed-forward layer of transformer block
      - *gpt.py*: the whole GPT architecture
      - *README.md*: notes about GPT as a whole and attention in particular
      - *transformer_block.py*: building block of GPT including self-attention and feed-forward
    - *generate.py*: code for generating new tokens with help of pre-trained model
    - *train.py*: code to train language model
    - *trainer.py*: code to do all the necessary step for training and evaluating the model
  - **utils**: various utils files
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
    1. Model name: `--model [bigram, gpt]` .
    2. Model size: `--size [small, large]` (small is good for debugging alongside with dataset fraction).
    3. Device: `--device [cpu, cuda, mps]` **[Optional]**: if not provided will try to detect automatically trying to use GPU first and if it's not available - fallback to cpu.
    4. Dataset fraction: `--dataset-fraction` **[Optional]**: usuful if one wants to quickly run training for debugging (affects both training and testing datasets). If not provided the whole dataset will be used.
    Arguments 1 and 2 are required.

    ```python
    python src/model/train.py --model gpt --size large
    ```

    ... or in one of the example notebooks in `notebooks/examples/*_model_training.ipynb`.

5. Run new token generation:

    Generation script accepts multiple arguments:
    1. Model name: `--model [bigram, gpt]` .
    2. Model size: `--size [small, large]` (small is good for debugging).
    3. Device: `--device [cpu, cuda, mps]` **[Optional]**: if not provided will try to detect automatically trying to use GPU first and if it's not available - fallback to cpu.
    4. Max new tokens: `--max-new-tokens` **[Optional]**: number of tokens to generate. If not provided the default value will be used, which is 100.

    ```python
    python src/model/generate.py --model gpt --size large --max-new-tokens 100
    ```

***

## Additional: git pre-commit hook

In order to run `black` formatter before each commit you need to add them into `.git/hooks` folder either manually or with helper script:

```bash
sh .add_git_hooks.sh`
```

This script will put `pre-commit` file into `.git/hooks` folder of the project and make it executable.
