from torch import Tensor, nn

from src.model import FeedForward, MultiHeadAttention


class TransformerBlock(nn.Module):
    def __init__(
        self,
        embeddings_size: int,
        head_size: int | None,
        num_heads: int,
        context_size: int,
        dropout: float,
        *,
        is_decoder: bool,
    ) -> None:
        """Create transformer block with self-attention, layer normalization and feed-forward.

        Self-attention is used in order to add communication between tokens, feed-forward - for
        processing this information. Layer normalization allows to build deeper neural networks.
        `Note`: pre-normalization of layers is used here.

        Parameters
        ----------
        embeddings_size : int
            size of the embeddings - the size of input of self-attention
        head_size : int | None
            the size of output of self-attention;
            if not provided `head_size` will be equal to `embeddings_size` // `num_heads`, so it should be divisible
            without residual
        num_heads : int
            how many self-attention heads to use
        context_size : int
            the number of tokens that will be used during calculation attention map and
            weighted averaging of value of each token
        dropout : float
            how many connection between tokens are dropped during each forward pass
        is_decoder : bool
            if it's a decoder masking of 'future' tokens will be applied
        """
        super().__init__()

        self.self_attention = MultiHeadAttention(
            context_size=context_size,
            dropout=dropout,
            head_size=head_size,
            embeddings_size=embeddings_size,
            num_heads=num_heads,
            is_decoder=is_decoder,
        )
        # TODO: make sure that `head_size` is 4
        # it shows that `head_size` is 32
        self.feed_forward = FeedForward(embeddings_size=embeddings_size, scaling=4, dropout=dropout)
        self.layer_norm_1 = nn.LayerNorm(embeddings_size)
        self.layer_norm_2 = nn.LayerNorm(embeddings_size)

    def forward(self, x: Tensor) -> Tensor:
        """Apply transformer block with layer norm, self-attention and feed-forward.

        `+` sign is for residual connection (allows to build deeper neural nets)

        Parameters
        ----------
        x : Tensor
            input tensor of size (batch_size, time-steps, channels_num)

        Returns
        -------
        Tensor
            output tensor of size (batch_size, time-steps, channels_num)
            output has the same size as input
        """
        # + sign is used for residual connection
        # helps with gradient flow and allows to build deeper neural nets
        x = x + self.self_attention(self.layer_norm_1(x))
        return x + self.feed_forward(self.layer_norm_2(x))
