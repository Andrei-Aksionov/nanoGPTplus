import torch
import torch.nn.functional as F
from torch import Tensor, nn


class Head(nn.Module):
    def __init__(
        self, embeddings_size: int, head_size: int, context_size: int, dropout: float, *, is_decoder: bool
    ) -> None:
        super().__init__()

        self.embeddings_size = embeddings_size
        self.head_size = head_size
        self.context_size = context_size
        self.is_decoder = is_decoder

        # what don't need `bias` because we simply want to do matrix multiplications
        self.key = nn.Linear(in_features=embeddings_size, out_features=head_size, bias=False)
        self.query = nn.Linear(in_features=embeddings_size, out_features=head_size, bias=False)
        self.value = nn.Linear(in_features=embeddings_size, out_features=head_size, bias=False)
        # If you have parameters in your model, which should be saved and restored in the state_dict,
        # but not trained by the optimizer, you should register them as buffers.
        # Buffers won’t be returned in model.parameters(), so that the optimizer won’t have a change to update them.
        self.register_buffer("tril", torch.tril(torch.ones(self.context_size, self.context_size)))

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: Tensor) -> Tensor:
        """
        Self-attention does these things:

        1. takes information from the token x
        2. encodes it into vectors key, query and value, where:
            a. key - information of what I have
            b. query - what I am looking for
            c. value - if you find me interesting that's what I represent
        3. creates triangular matrix that will help to mask tokens from the future, in other words
        token at position 4 should not communicate with token at position 5 and above, only with the previous tokens
        (3 and below)
        4. mask token from the future with -inf value, which after softmax operation becomes 0
        5. does weighted sum by multiplying obtained scores and value matrix
        """

        b, t, c = x.shape

        # key - what do I represent
        # query - what I am looking for
        k: Tensor = self.key(x)  # (B, T, C)
        q: Tensor = self.query(x)  # (B, T, C)

        # compute attention scores ("affinities")
        wei: Tensor = q @ k.transpose(-2, -1)  # (B, T, C) @ (B, C, T) -> (B, T, T)
        # or
        # wei = q @ k.mT

        # TODO: write down why to do that
        """
        In order to preserve 1 unit variance of the product of multiplication of two vectors
        we need to divide by square root of the features size
        We need it to make sure that the values after softmax are well spread out, otherwise in worst
        case scenario the values after the softmax will converge to one-hot encoding (like [0, 0, 1]) and
        that will mean that the attention will be on a single (or couple of) tokens, and we want it to be
        spread out (like [0.2, 0.1, 0.7])
        we want to aggregate information not from a single node
        """
        wei *= k.shape[-1] ** -0.5

        # TODO: in the original code wei was scaled by c (number of channels)
        # now it is scaled by head size
        # figure out why it was changed
        if k.shape[-1] != c:
            msg = f"k of size '{k.shape}' is not equal to c of size '{c}'"
            # raise ValueError(msg)

        # TODO: masking is only needed for decoder part of transformer
        # I think it's better to move it out of this function in order to make
        # head be more general
        # or simple if-statement

        # [0.9, -0.6, 0.3]    [0.9, -inf, -inf]
        # [0.1, 0.5, -0.1] -> [0.1, 0.5, -inf]
        # [0.1, 0.2, 0.3]     [0.1, 0.2, 0.3]
        # and after softmax -inf becomes 0
        # this doesn't allow current token communicate with future ones
        if self.is_decoder:
            wei = wei.masked_fill(self.tril[:t, :t] == 0, float("-inf"))  # (B, T, T)

        wei = F.softmax(wei, dim=-1)  # (B, T, T)

        # randomly prevent some nodes from communicating, some of theme randomly are set to zero
        # helps prevent overfitting
        wei = self.dropout(wei)

        # perform the weighted aggregation of the values
        v: Tensor = self.value(x)  # (B, T, C)
        out = wei @ v  # (B, T, T) @ (B, T, C) -> (B, T, C)
        return out


class MultiHeadAttention(nn.Module):
    """
    Multi head attention is simply applying multiple attentions in parallel and concatenating results.
    # TODO: write how different heads might learn different scale, some of them might learn short range attention,
    # while the others might focus on long range ones.

    It creates multiple independent channels of communication, gather a lot of different data.
    """

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        embeddings_size: int,
        dropout: float,
        context_size: int,
        *,
        is_decoder: bool,
    ) -> None:
        super().__init__()

        # number of heads to be run in parallel
        self.num_heads = num_heads
        # the size of each head
        self.head_size = head_size
        self.embeddings_size = embeddings_size
        self.context_size = context_size

        self.heads = nn.ModuleList(
            [
                Head(
                    embeddings_size=self.embeddings_size,
                    head_size=self.head_size,
                    context_size=self.context_size,
                    dropout=dropout,
                    is_decoder=is_decoder,
                )
                for _ in range(self.num_heads)
            ]
        )

        # TODO: why do we need projection in the first place?
        # We need projection only if the size of dimension is larger than x, because we need to have the
        # same size for performing residual connection
        # In my opinion if the size is not changed we might skip it
        # self.projection = nn.Linear(self.n_embed, self.n_embed)
        self.projection = nn.Linear(self.head_size * self.num_heads, self.embeddings_size)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: Tensor) -> Tensor:
        # concatenate over channel dimension
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.projection(out)
        return self.dropout(out)
