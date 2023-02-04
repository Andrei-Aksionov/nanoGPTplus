import torch
import torch.nn.functional as F
from torch import Tensor, nn

from src.model.gpt_lm.attention import MultiHeadAttention
from src.utils.device import get_device


class FeedForward(nn.Module):
    """
    Applied on per-token level. Each token is processed independently.

    # TODO: find info from the deep dive book
    As I understand after self-attention each token has attention map (?) and now
    with help of simple fully connected layers this data is processed

    The question: what is exactly is stored in vectors for each token after self-attention
    Attention scores are multiplied on value matrix and the result is what?

    In the video Andrej noted that attention step is for communication, and feed-forward
    is for computation.


    """

    def __init__(self, n_embed: int, scaling: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embed, scaling * n_embed),
            nn.ReLU(),
            nn.Linear(scaling * n_embed, n_embed),
            nn.Dropout(p=dropout),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class Block(nn.Module):
    def __init__(self, n_embed, block_size, dropout, num_heads) -> None:
        super().__init__()
        head_size = n_embed // num_heads
        self.self_attention = MultiHeadAttention(
            block_size=block_size,
            dropout=dropout,
            head_size=head_size,
            n_embed=n_embed,
            num_heads=num_heads,
        )
        # TODO: make sure that `head_size` is 4
        # it shows that `head_size` is 32
        # print(f"{head_size=}")
        # self.feed_forward = FeedForward(n_embed, head_size, dropout)
        self.feed_forward = FeedForward(n_embed=n_embed, scaling=4, dropout=dropout)
        self.layer_norm_1 = nn.LayerNorm(n_embed)
        self.layer_norm_2 = nn.LayerNorm(n_embed)

    def forward(self, x: Tensor) -> Tensor:
        # TODO: check why in-place doesn't work during backpropagation
        # x += self.self_attention(self.layer_norm_1(x))
        # x += self.feed_forward(self.layer_norm_2(x))

        # + sign is used for residual connection
        # helps with gradient flow and allows to build deeper neural nets
        x = x + self.self_attention(self.layer_norm_1(x))
        x = x + self.feed_forward(self.layer_norm_2(x))
        return x


class GPT(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        n_embed: int,
        n_layers: int,
        block_size: int,
        dropout: float,
        num_heads: int,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.n_embed = n_embed
        self.n_layer = n_layers
        self.block_size = block_size
        self.device = get_device()

        self.token_embedding_table = nn.Embedding(num_embeddings=self.vocab_size, embedding_dim=self.n_embed)
        # since attention doesn't have any notion of space and we want to use spatial information we need to implement
        # positional embeddings (they will encode relative position of each token)
        # positional embeddings knows how to encode position of last N (block_size) tokens
        self.positional_embedding_table = nn.Embedding(num_embeddings=self.block_size, embedding_dim=self.n_embed)
        self.blocks = nn.Sequential(
            *[
                Block(n_embed=n_embed, block_size=block_size, dropout=dropout, num_heads=num_heads)
                for _ in range(self.n_layer)
            ],
        )
        # TODO: why `normalized_shape` is equal to `n_embed`
        # LayerNorm - normalizes features for each sample independently
        self.layer_norm = nn.LayerNorm(n_embed)  # final layer norm
        self.language_model_head = nn.Linear(n_embed, vocab_size)

    def forward(self, idx: Tensor) -> Tensor:
        b, t = idx.shape
        token_embeddings = self.token_embedding_table(idx)  # (B, T, C)
        # TODO: do we need to create range of length t each time in feed forward method,
        # since t (number of time steps) is predefined?
        positional_embeddings = self.positional_embedding_table(torch.arange(t, device=self.device))  # (T, C)
        x = token_embeddings + positional_embeddings  # (B, T, C)
        x = self.blocks(x)  # (B, T, C)
        x = self.layer_norm(x)  # (B, T, C)
        logits = self.language_model_head(x)  # (B, T, vocab_size)

        return logits

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
            context = idx[:, -self.block_size :]
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
