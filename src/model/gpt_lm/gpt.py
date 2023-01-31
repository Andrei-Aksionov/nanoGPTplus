import torch
import torch.nn.functional as F
from torch import Tensor, nn

from src.utils.device import get_device


class GPT(nn.Module):
    def __init__(self, vocab_size: int, n_embed: int, n_layers: int, block_size: int) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.n_embed = n_embed
        self.n_layer = n_layers
        self.block_size = block_size
        self.device = get_device()

        self.token_embedding_table = nn.Embedding(num_embeddings=self.vocab_size, embedding_dim=self.n_embed)
        self.positional_embedding_table = nn.Embedding(num_embeddings=self.block_size, embedding_dim=self.n_embed)

    def forward(self, idx: Tensor) -> Tensor:
        b, t = idx.shape
        token_embeddings = self.token_embedding_table(idx)
        positional_embeddings = self.positional_embedding_table(torch.arange(t, device=self.device))
        return token_embeddings + positional_embeddings

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
        # contains specific loss to the bigram language model
        b, t, c = logits.shape
        return F.cross_entropy(
            logits.view(b * t, c),
            targets.view(b * t),
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
