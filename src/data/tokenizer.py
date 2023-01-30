class CharTokenizer:
    def __init__(self, vocab: list[str] = None, corpus: str = None) -> None:
        """Create tokenizer.

        If vocab is not provided it will generate vocab from the corpus.
        So either of them should be provided.

        Parameters
        ----------
        vocab : list[str], optional
            list of unique tokens (chars, words, sub-words, ...), by default None
        corpus : str, optional
            the whole text on which model will be trained, required to generate vocabulary if not provided,
            by default None
        """
        assert corpus or vocab, "Either corpus or vocabulary has to be provided"

        if not vocab:
            self.vocab = set(corpus)
        self.vocab = sorted(self.vocab)
        self.vocab_size = len(self.vocab)
        # create mapping from characters to integers and inverse
        self.stoi = {char: idx for idx, char in enumerate(self.vocab)}
        self.itos = {idx: char for idx, char in enumerate(self.vocab)}

    def encode(self, text: str) -> list[int]:
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

    def decode(self, indices: list[int]) -> str:
        """Decode list of indices into a string.

        Parameters
        ----------
        indices : list[int]
            each integer corresponds to the index in the vocabulary

        Returns
        -------
        str
            decoded text
        """
        return "".join(self.itos[idx] for idx in indices)
