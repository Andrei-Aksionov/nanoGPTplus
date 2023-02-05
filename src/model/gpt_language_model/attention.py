import torch
import torch.nn.functional as F
from torch import Tensor, nn


class SelfAttentionHead(nn.Module):
    def __init__(
        self,
        embeddings_size: int,
        head_size: int,
        context_size: int,
        dropout: float,
        *,
        is_decoder: bool,
    ) -> None:
        """Single self-attention head.

        Parameters
        ----------
        embeddings_size : int
            size of the embeddings - the size of input of self-attention
        head_size : int
            the size of output of self-attention
        context_size : int
            the number of tokens that will be used during calculation attention map and
            weighted averaging of value of each token
        dropout : float
            how many connection between tokens are dropped during each forward pass
        is_decoder : bool
            if it's a decoder masking of 'future' tokens will be applied
        """
        super().__init__()

        self.embeddings_size = embeddings_size
        self.head_size = head_size
        self.context_size = context_size
        self.is_decoder = is_decoder

        # what don't need `bias` because we simply want to do matrix multiplications
        self.key_weights = nn.Linear(embeddings_size, head_size, bias=False)
        self.query_weights = nn.Linear(embeddings_size, head_size, bias=False)
        self.value_weights = nn.Linear(embeddings_size, head_size, bias=False)
        # If you have parameters in your model, which should be saved and restored in the state_dict,
        # but not trained by the optimizer, you should register them as buffers.
        # Buffers won't be returned in model.parameters(), so that the optimizer won't have a change to update them.
        self.register_buffer("tril", torch.tril(torch.ones(self.context_size, self.context_size)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        """Forward method for Self-Attention.

        Self-attention does these 6 steps:

        1. takes information from the token x
        2. encodes it into vectors key, query and value, where (intuitively):
            a. key - information of what the token has
            b. query - what the token is interested in
            c. value - if some token finds current token 'interesting' then this value should be used
        3. calculate attention scores by doing dot product between key and query
        4. [optional for decoder] creates triangular matrix that will help to mask tokens from the 'future', in other
        words token at position 4 should not communicate with token at position 5 and above, only with the previous
        tokens (3 and below)
        5. mask token from the 'future' with -inf value, which after softmax operation becomes 0
        6. does weighted sum by multiplying obtained attention scores and value matrix
        """
        # batch, timestep, channels
        b, t, c = x.shape

        key = self.key_weights(x)  # (B, T, C)
        query = self.query_weights(x)  # (B, T, C)
        value = self.value_weights(x)  # (B, T, C)

        # first to obtaining attention scores: dot product of key and query
        attention_scores = query @ key.mT  # (B, T, C) @ (B, C, T) -> (B, T, T)

        # In order to preserve 1 unit variance of the product of multiplication of two vectors
        # we need to divide by square root of the features size (in our case - attention head size)
        # We need it to make sure that the values after softmax are well spread out, otherwise in worst
        # case scenario the values after the softmax will converge to one-hot encoding (like [0, 0, 1]) and
        # that will mean that the attention will be on a single (or couple of) tokens, and we want it to be
        # spread out (like [0.2, 0.1, 0.7])
        # we want to aggregate information not from a single node
        head_size = key.shape[-1]
        attention_scores *= head_size**-0.5

        # if it's a decoder we need to mask 'future' tokens with '-inf' value
        if self.is_decoder:
            # [0.9, -0.6, 0.3] -> [0.9, -inf, -inf]
            # [0.1, 0.5, -0.1] -> [0.1, 0.5, -inf]
            # [0.1, 0.2, 0.3]  -> [0.1, 0.2, 0.3]
            # and after softmax -inf becomes 0
            # this doesn't allow current token communicate with future ones
            attention_scores = attention_scores.masked_fill(self.tril[:t, :t] == 0, float("-inf"))  # (B, T, T)

        # since we want to do weighted averaging we need to transform attention scores into range [0, 1]
        # and sum of all scores should be equal to 1; softmax is a good tool for it
        attention_scores = F.softmax(attention_scores, dim=-1)  # (B, T, T)

        # randomly prevent some nodes from communicating, some of theme randomly are set to zero
        # helps prevent overfitting
        attention_scores = self.dropout(attention_scores)

        # perform the weighted aggregation of the values
        return attention_scores @ value  # (B, T, T) @ (B, T, C) -> (B, T, C)


class MultiHeadAttention(nn.Module):
    """Multiple Attentions running in parallel.

    Multi head attention is simply applying multiple attentions in parallel and concatenating results.
    # TODO: write how different heads might learn different scale, some of them might learn short range attention,
    # while the others might focus on long range ones.

    It creates multiple independent channels of communication, gather a lot of different data.
    """

    def __init__(
        self,
        embeddings_size: int,
        head_size: int,
        num_heads: int,
        context_size: int,
        dropout: float,
        *,
        is_decoder: bool,
    ) -> None:
        """Multiple self-attention heads run in parallel.

        Multi head attention is simply applying multiple self-attention heads in parallel and concatenation results.
        That helps to add more parallelization plus each head has it's own set of K, Q an V weights, which means
        that each head might learn something new; it allows the model to focus on different positions, maybe
        one head will focus more on near tokens, while others will focus on tokens that are far away.

        Parameters
        ----------
        embeddings_size : int
            size of the embeddings - the size of input of self-attention
        head_size : int
            the size of output of self-attention
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

        self.embeddings_size = embeddings_size
        self.head_size = head_size
        self.num_heads = num_heads
        self.context_size = context_size
        self.dropout = dropout
        self.is_decoder = is_decoder

        self.heads = nn.ModuleList(
            [
                SelfAttentionHead(
                    self.embeddings_size,
                    self.head_size,
                    self.context_size,
                    self.dropout,
                    is_decoder=self.is_decoder,
                )
                for _ in range(self.num_heads)
            ],
        )

        # if after concatenation the size of channels is bigger than embeddings size
        # projection will scaled it down
        self.projection = nn.Linear(self.head_size * self.num_heads, self.embeddings_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:  # noqa: D102
        # concatenate over channel dimension
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.projection(out)
        return self.dropout(out)
