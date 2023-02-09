import torch
import torch.nn.functional as F
from torch import Tensor, nn

from src.model.gpt_language_model.transformer_block import TransformerBlock


class GPTLanguageModel(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embeddings_size: int,
        context_size: int,
        head_size: int | None,
        num_heads: int,
        feed_forward_scaling: int,
        num_layers: int,
        dropout: float,
    ) -> None:
        """Create Generative Pre-trained Transformer model (decoder part of transformer architecture).

        Parameters
        ----------
        vocab_size : int
            number of unique tokens in vocabulary
        embeddings_size : int
            size of the embeddings - the size of input of self-attention
        context_size : int
            the number of tokens that will be used during calculation attention map and
            weighted averaging of value of each token
        head_size : int | None
            the size of output of self-attention
        num_heads : int
            how many self-attention heads to use
        feed_forward_scaling : int
            feed-forward has two fully-connected layers; the number of neurons between them is larger
            than input and output sizes, `feed_forward_scaling` specifies by how much
        num_layers : int
            how many transformer blocks to use
        dropout : float
            how many connection between tokens are dropped during each forward pass
        """
        super().__init__()

        self.vocab_size = vocab_size
        self.embeddings_size = embeddings_size
        self.context_size = context_size
        self.head_size = head_size
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.dropout = dropout
        self.feed_forward_scaling = feed_forward_scaling

        self.token_embedding_table = nn.Embedding(self.vocab_size, self.embeddings_size)
        # since attention doesn't have any notion of space and we want to use spatial information we need to implement
        # positional embeddings (they will encode relative position of each token)
        # positional embeddings knows how to encode position of last N (context_size) tokens
        self.positional_embedding_table = nn.Embedding(self.context_size, self.embeddings_size)
        self.positional_indices = torch.arange(self.context_size)
        self.embeddings_dropout = nn.Dropout(dropout)
        self.blocks = nn.Sequential(
            *[
                TransformerBlock(
                    embeddings_size=self.embeddings_size,
                    context_size=self.context_size,
                    head_size=self.head_size,
                    num_heads=self.num_heads,
                    dropout=self.dropout,
                    feed_forward_scaling=self.feed_forward_scaling,
                    is_decoder=True,
                )
                for _ in range(self.num_layers)
            ],
        )
        self.layer_norm = nn.LayerNorm(self.embeddings_size)  # final layer norm
        self.language_model_head = nn.Linear(self.embeddings_size, self.vocab_size)

        self.apply(self.__init_weights)

    def __init_weights(self, module: "torch.nn.modules") -> None:
        """Initialize Embedding and Linear layers with a smaller std.

        By default weights of Embedding layer are initialized from normal distribution
        with zero mean and unit std ( N(0, 1) ).
        Weights for linear layer are initialized from uniform distribution from
        [-k**0.5, k**0.5], where k = 1 / in_features. Even with 128 features it will be
        [-0.09, 0.09].

        Parameters
        ----------
        module : torch.nn.modules
            module of the network
        """
        if isinstance(module, (nn.Embedding, nn.Linear)):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if hasattr(module, "bias") and module.bias is not None:
                torch.nn.init.zeros_(module.bias)

    def forward(self, idx: Tensor) -> Tensor:
        """Do the whole forward pass for decoder part of transformer.

        This forward method includes all steps for decoder:
        1. token embeddings + positional
        2. transformer block consisting of self-attention, feed-forward, addNorm
        3. logits for each token in vocabulary

        Parameters
        ----------
        idx : Tensor
            tensor of size (batch, time-step) consisting of indices of tokens inside vocabulary
            for each time-step for each batch

        Returns
        -------
        Tensor
            tensor of size (batch, time-step, vocabulary_size): logits for each token in vocabulary
            for each time-step for each batch
        """
        # batch, time-step
        B, T = idx.shape  # noqa: N806
        # obtain token embeddings and add positional information
        token_embeddings = self.token_embedding_table(idx)  # (B, T, C)
        positional_embeddings = self.positional_embedding_table(self.positional_indices[:T])  # (T, C)
        x = token_embeddings + positional_embeddings  # (B, T, C)
        x = self.embeddings_dropout(x)
        # apply multiple transformer blocks
        x = self.blocks(x)  # (B, T, C)
        # apply final normalization and generate logits for each token in vocabulary
        x = self.layer_norm(x)  # (B, T, C)
        return self.language_model_head(x)  # (B, T, vocab_size)

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
