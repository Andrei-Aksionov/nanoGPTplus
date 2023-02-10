import torch
from torch import Tensor, nn
from torch.nn import functional as F


class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size: int) -> None:
        """Bigram Language Model that predicts most likely next character.

        The next character is generated by sampling distribution of relative frequency, that shows
        what is most frequent character after the current one.

        Parameters
        ----------
        vocab_size : int
            required to properly build square embedding layer, where each row corresponds to unique character
            from the vocabulary and each column -> "frequency" of the next character.
        """
        super().__init__()
        # each token directly reads off the logits for the next token from the lookup table
        # in other words for each token "feature" is the "frequency/probability" for each next word
        # that why it has square size
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx: Tensor) -> Tensor:  # noqa: D102
        # if don't want to deal with custom loss, one can run '.mT' method of the tensor like:
        # > self.token_embedding_table(idx).mT
        # though it's slower
        return self.token_embedding_table(idx)

    def loss(self, logits: Tensor, targets: Tensor) -> Tensor:
        """Prepare tensors to be compatible with Pytorch's Cross-Entropy loss and applies it.

        Cross-Entropy expects to have features as the second-dimension, so we first need to
        transform logits in supported shape and in order to align targets tensor with logits tensor,
        we need to transform targets too.

        Parameters
        ----------
        logits : Tensor
            tensor with model's outputs
        targets : Tensor
            tensor with true labels

        Returns
        -------
        Tensor
            tensor with loss value (of how good model's predictions are)
        """
        B, T, C = logits.shape  # noqa: N806
        return F.cross_entropy(
            logits.view(B * T, C),
            targets.view(B * T),
        )

    def generate(self, idx: Tensor, max_new_tokens: int) -> Tensor:
        """Generate new character after the current one.

        Parameters
        ----------
        idx : Tensor
            index of the current character
        max_new_tokens : int
            number of characters to be generated

        Returns
        -------
        Tensor
            tensor containing indices of the provided characters and newly generated
        """
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # get the predictions
            logits = self(idx)  # (B, T, C)
            # focus only on the last time step
            logits = logits[:, -1, :]  # becomes (B, C)
            # apply softmax on the predictions to get probabilities
            probs = F.softmax(logits, dim=-1)  # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T + 1)
        return idx