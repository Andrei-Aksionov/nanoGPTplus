from torch import Tensor, nn


class FeedForward(nn.Module):
    def __init__(self, embeddings_size: int, bias: bool, scaling: int, dropout: float) -> None:
        """Apply on per-token level. Each token is processed independently.

        If the is no feed-forward layer, self-attention is simply a process of re-averaging of value vectors. In order
        to add element-wise non-linearity transformation of incoming vectors we add feed-forward part.

        You can think about it in this way:
        - attention step is for communication between tokens
        - feed-forward is for processing this information (of how tokens are related to each other via attention)

        Parameters
        ----------
        embeddings_size : int
            size of the embeddings - the size of input of self-attention
        bias: bool
            whether to use bias or not: without bias might be a bit better and faster
        scaling : int
            feed-forward has two fully-connected layers; the number of neurons between them is larger
            than input and output sizes, `scaling` specifies by how much
        dropout : float
            how many connection between tokens are dropped during each forward pass
        """
        super().__init__()

        self.embeddings_size = embeddings_size
        self.bias = bias
        self.scaling = scaling
        self.dropout = dropout

        self.net = nn.Sequential(
            nn.Linear(self.embeddings_size, self.scaling * self.embeddings_size, bias=self.bias),
            # TODO: try GELU
            nn.ReLU(),
            nn.Linear(self.scaling * self.embeddings_size, self.embeddings_size, bias=self.bias),
            nn.Dropout(self.dropout),
        )

    def forward(self, x: Tensor) -> Tensor:  # noqa: D102
        return self.net(x)
