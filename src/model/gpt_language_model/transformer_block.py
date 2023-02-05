from torch import Tensor, nn

from src.model import FeedForward, MultiHeadAttention


class TransformerBlock(nn.Module):
    def __init__(self, embeddings_size, context_size, dropout, num_heads, is_decoder, head_size=None) -> None:
        super().__init__()

        if not head_size:
            # TODO: write proper exception
            assert embeddings_size % num_heads == 0, "Cannot do integer division"
            head_size = embeddings_size // num_heads

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
        # + sign is used for residual connection
        # helps with gradient flow and allows to build deeper neural nets
        x = x + self.self_attention(self.layer_norm_1(x))
        return x + self.feed_forward(self.layer_norm_2(x))
