from torch import Tensor, nn


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

    Consider encoder part of transformer.  If there is no feed-forward layer, self-attention is simply performing
    re-averaging of value vectors.  In order to add more model function, i.e. element-wise non-linearity transformation
    of incoming vectors, to transformer, we add feed-forward layer to encoder part of transformer.

    """

    def __init__(self, embeddings_size: int, scaling: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embeddings_size, scaling * embeddings_size),
            nn.ReLU(),
            nn.Linear(scaling * embeddings_size, embeddings_size),
            nn.Dropout(p=dropout),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)
