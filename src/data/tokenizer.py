class CharTokenizer:
    def __init__(self, text: str) -> None:
        # get all unique characters and count them
        self.vocab = sorted(set(text))
        self.vocab_size = len(self.vocab)

        # create mapping from characters to integers and inverse
        self.stoi = {char: idx for idx, char in enumerate(self.vocab)}
        self.itos = dict(enumerate(self.vocab))

    def encode(self, text: str) -> list[int]:
        return [self.stoi[ch] for ch in text]

    def decode(self, indices: list[int]) -> str:
        return "".join(self.itos[idx] for idx in indices)
