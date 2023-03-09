import pytest

from src import config
from src.model import BigramLanguageModel, GPTLanguageModel
from src.utils import grab_arguments


@pytest.mark.smoke
@pytest.mark.order(2)
class TestModel:
    @classmethod
    def setup_class(cls: "TestModel") -> None:
        # vocab size is derived after text tokenization
        # so for these tests it's totally arbitrary
        cls.vocab_size = 10

    def test_bigram_configs_load(self) -> None:
        _ = BigramLanguageModel(vocab_size=TestModel.vocab_size)

    @pytest.mark.parametrize("model_size", list(config.model.gpt.size.keys()))
    def test_gpt_configs_load(self, model_size: str) -> None:
        _ = GPTLanguageModel(
            vocab_size=TestModel.vocab_size,
            **grab_arguments(GPTLanguageModel, config.model.gpt.size[model_size]),
        )
