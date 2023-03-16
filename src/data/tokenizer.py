from typing import List, Optional


class CharTokenizer:
    def __init__(self, vocab: Optional[List[str]] = None, corpus: Optional[str] = None) -> None:
        """Create tokenizer.

        If vocab is not provided it will generate vocab from the corpus.
        So either of them should be provided.

        Parameters
        ----------
        vocab : Optional[list[str]], optional
            list of unique tokens (chars, words, sub-words, ...), by default None
        corpus : Optional[str], optional
            the whole text on which model will be trained, required to generate vocabulary if not provided,
            by default None

        Raises
        ------
        ValueError
            raises error if neither corpus nor vocabulary was provided
        """
        if not (corpus or vocab):
            raise ValueError("Neither corpus nor vocabulary is provided")

        if not vocab:
            self.vocab = set(corpus)
        self.vocab = sorted(self.vocab)
        self.vocab_size = len(self.vocab)
        # create mapping from characters to integers and inverse
        self.stoi = {char: idx for idx, char in enumerate(self.vocab)}
        self.itos = {idx: char for idx, char in enumerate(self.vocab)}

    def encode(self, text: str) -> List[int]:
        """Encode text input into corresponding list of indices from char->idx map.

        Parameters
        ----------
        text : str
            any text

        Returns
        -------
        list[int]
            each integer corresponds to the index in the vocabulary
        """
        return [self.stoi[ch] for ch in text]

    def decode(self, indices: List[int]) -> str:
        """Decode list of indices into a string.

        Parameters
        ----------
        indices : List[int]
            each integer corresponds to the index in the vocabulary

        Returns
        -------
        str
            decoded text
        """
        return "".join(self.itos[idx] for idx in indices)
