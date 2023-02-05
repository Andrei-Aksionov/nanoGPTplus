import torch
import torch.nn.functional as F
from torch import Tensor, nn

from src import config
from src.model.gpt_language_model.transformer_block import TransformerBlock
from src.utils.arguments import grab_arguments
from src.utils.device import get_device


class GPTLanguageModel(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embeddings_size: int,
        num_layers: int,
        context_size: int,
        dropout: float,
        num_heads: int,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.embeddings_size = embeddings_size
        self.num_layers = num_layers
        self.context_size = context_size
        self.device = get_device()

        self.token_embedding_table = nn.Embedding(num_embeddings=self.vocab_size, embedding_dim=self.embeddings_size)
        # since attention doesn't have any notion of space and we want to use spatial information we need to implement
        # positional embeddings (they will encode relative position of each token)
        # positional embeddings knows how to encode position of last N (block_size) tokens
        self.positional_embedding_table = nn.Embedding(
            num_embeddings=self.context_size, embedding_dim=self.embeddings_size
        )
        self.blocks = nn.Sequential(
            *[
                TransformerBlock(
                    **grab_arguments(TransformerBlock, config.model.gpt.size.small),
                    is_decoder=True,
                )
                for _ in range(self.num_layers)
            ],
        )
        # TODO: why `normalized_shape` is equal to `n_embed`
        # LayerNorm - normalizes features for each sample independently
        self.layer_norm = nn.LayerNorm(embeddings_size)  # final layer norm
        self.language_model_head = nn.Linear(embeddings_size, vocab_size)

    def forward(self, idx: Tensor) -> Tensor:
        b, t = idx.shape  # (32, 8)
        token_embeddings = self.token_embedding_table(idx)  # (B, T, C) (32, 8, 64)
        # TODO: do we need to create range of length t each time in feed forward method,
        # since t (number of time steps) is predefined?
        positional_embeddings = self.positional_embedding_table(torch.arange(t, device=self.device))  # (T, C) (8, 64)
        x = token_embeddings + positional_embeddings  # (B, T, C) (32, 8, 64)
        x = self.blocks(x)  # (B, T, C) (32, 8, 64)
        x = self.layer_norm(x)  # (B, T, C) (32, 8, 64)
        return self.language_model_head(x)  # (B, T, vocab_size) (32, 8, 65)

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

    @torch.no_grad()
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
            # crop idx to the last block_size items
            context = idx[:, -self.context_size :]
            # get the predictions
            logits = self(context)  # (B, T, C)
            # focus only on the last time step
            logits = logits[:, -1, :]  # becomes (B, C)
            # apply softmax on the predictions to get probabilities
            probs = F.softmax(logits, dim=-1)  # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T + 1)
        return idx
