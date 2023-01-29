import torch
import torch.nn as nn
from torch.nn import functional as F


class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size) -> None:
        super().__init__()
        # each token directly reads off the logits for the next token from the lookup table
        # in other words for each token "feature" is the "frequency/probability" for each next word
        # that why it has square size
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx) -> torch.Tensor:
        # if don't want to deal with custom loss, one can run '.mT' method of the tensor like:
        # > self.token_embedding_table(idx).mT
        # though it's slower
        return self.token_embedding_table(idx)

    def loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Prepares tensors to be compatible with Pytorch's Cross-Entropy loss and applies it.

        Cross-Entropy expects to have features as the second-dimension, so we first need to
        transform logits in supported shape and in order to align targets tensor with logits tensor,
        we need to transform targets too.

        Parameters
        ----------
        logits : torch.Tensor
            tensor with model's outputs
        targets : torch.Tensor
            tensor with true labels

        Returns
        -------
        torch.Tensor
            tensor with loss value (of how good model's predictions are)
        """
        # contains specific loss to the bigram language model
        B, T, C = logits.shape
        return F.cross_entropy(
            logits.view(B * T, C),
            targets.view(B * T),
        )

    def generate(self, idx, max_new_tokens):
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
