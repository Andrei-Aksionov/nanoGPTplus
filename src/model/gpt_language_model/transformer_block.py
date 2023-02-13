from torch import Tensor, nn

from src.model.gpt_language_model.attention import (
    CausalSelfAttention,
    MultiHeadAttention,
)
from src.model.gpt_language_model.feed_forward import FeedForward


# TODO: remove line below
# flake8: noqa
# cSpell:disable
class LayerNorm(nn.LayerNorm):
    # def __init__(self, normalized_shape: _shape_t, eps: float = 0.00001, elementwise_affine: bool = True, device=None, dtype=None) -> None:
    #     super().__init__(normalized_shape, eps, elementwise_affine, device, dtype)
    def __init__(self, normalized_shape, bias, **kwargs) -> None:
        super().__init__(normalized_shape, **kwargs)
        if not bias:
            self.bias = None


# TODO: remove line below
# cSpell:enable


class TransformerBlock(nn.Module):
    def __init__(
        self,
        embeddings_size: int,
        context_size: int,
        head_size: int | None,
        num_heads: int,
        bias: bool,
        dropout: float,
        feed_forward_scaling: int,
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
        context_size : int
            the number of tokens that will be used during calculation attention map and
            weighted averaging of value of each token
        head_size : int | None
            the size of output of self-attention;
            if not provided `head_size` will be equal to `embeddings_size` // `num_heads`, so it should be divisible
            without residual
        num_heads : int
            how many self-attention heads to use
        bias: bool
            whether to use bias or not: without bias might be a bit better and faster
        dropout : float
            how many connection between tokens are dropped during each forward pass
        feed_forward_scaling: int
            feed-forward has two fully-connected layers; the number of neurons between them is larger
            than input and output sizes, `feed_forward_scaling` specifies by how much
        is_decoder : bool
            if it's a decoder masking of 'future' tokens will be applied
        """
        super().__init__()

        self.embeddings_size = embeddings_size
        self.context_size = context_size
        self.head_size = head_size
        self.num_heads = num_heads
        self.bias = bias
        self.dropout = dropout
        self.feed_forward_scaling = feed_forward_scaling
        self.is_decoder = is_decoder

        # self.self_attention = MultiHeadAttention(
        #     embeddings_size=self.embeddings_size,
        #     context_size=self.context_size,
        #     head_size=self.head_size,
        #     num_heads=self.num_heads,
        #     dropout=self.dropout,
        #     is_decoder=self.is_decoder,
        # )
        self.self_attention = CausalSelfAttention(
            embeddings_size=self.embeddings_size,
            context_size=self.context_size,
            head_size=self.head_size,
            num_heads=self.num_heads,
            bias=self.bias,
            dropout=self.dropout,
        )
        self.feed_forward = FeedForward(
            embeddings_size=self.embeddings_size,
            bias=self.bias,
            scaling=self.feed_forward_scaling,
            dropout=self.dropout,
        )
        self.layer_norm_1 = LayerNorm(normalized_shape=self.embeddings_size, bias=self.bias)
        self.layer_norm_2 = LayerNorm(normalized_shape=self.embeddings_size, bias=self.bias)

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
