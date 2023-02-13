import torch
import torch.nn.functional as F
from loguru import logger
from torch import Tensor, nn


class SelfAttentionHead(nn.Module):
    def __init__(
        self,
        embeddings_size: int,
        context_size: int,
        head_size: int,
        dropout: float,
        *,
        is_decoder: bool,
    ) -> None:
        """Single self-attention head.

        Parameters
        ----------
        embeddings_size : int
            size of the embeddings - the size of input of self-attention
        context_size : int
            the number of tokens that will be used during calculation attention map and
            weighted averaging of value of each token
        head_size : int
            the size of output of self-attention
        dropout : float
            how many connection between tokens are dropped during each forward pass
        is_decoder : bool
            if it's a decoder masking of 'future' tokens will be applied
        """
        super().__init__()

        self.embeddings_size = embeddings_size
        self.context_size = context_size
        self.head_size = head_size
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

        Parameters
        ----------
        x : Tensor
            input tensor containing vector representation of x

        Returns
        -------
        Tensor
            output vector representation of x of size (batch, time-step, head_size)
        """
        # batch, time-step, channels
        B, T, _ = x.shape  # noqa: N806

        # here and below `C` means `head_size`, not channels of input
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
            # and after softmax `-inf` becomes 0
            # this doesn't allow current token communicate with future ones
            attention_scores = attention_scores.masked_fill(self.tril[:T, :T] == 0, float("-inf"))  # (B, T, T)

        # since we want to do weighted averaging we need to transform attention scores into range [0, 1]
        # and sum of all scores should be equal to 1; softmax is a good tool for it
        attention_scores = F.softmax(attention_scores, dim=-1)  # (B, T, T)

        # randomly prevent some nodes from communicating, some of theme randomly are set to zero
        # helps prevent overfitting
        attention_scores = self.dropout(attention_scores)

        # perform the weighted aggregation of the values
        return attention_scores @ value  # (B, T, T) @ (B, T, C) -> (B, T, C)


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        embeddings_size: int,
        context_size: int,
        head_size: int | None,
        num_heads: int,
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
        context_size : int
            the number of tokens that will be used during calculation attention map and
            weighted averaging of value of each token
        head_size : int | None
            the size of output of self-attention;
            if not provided `head_size` will be equal to `embeddings_size` // `num_heads`, so it should be divisible
            without residual
        num_heads : int
            how many self-attention heads to use
        dropout : float
            how many connection between tokens are dropped during each forward pass
        is_decoder : bool
            if it's a decoder masking of 'future' tokens will be applied

        Raises
        ------
        ValueError
            if `embeddings_size` cannot be divided by `num_heads` without residual
        """
        super().__init__()

        if not head_size:
            if embeddings_size % num_heads != 0:
                msg = "Embeddings size should be divisible by number of heads without residual, "
                f"but was provided: embeddings_size={embeddings_size}; num_heads={num_heads}"
                raise ValueError(msg)
            head_size = embeddings_size // num_heads

        self.embeddings_size = embeddings_size
        self.context_size = context_size
        self.head_size = head_size
        self.num_heads = num_heads
        self.dropout = dropout
        self.is_decoder = is_decoder

        self.heads = nn.ModuleList(
            [
                SelfAttentionHead(
                    embeddings_size=self.embeddings_size,
                    context_size=self.context_size,
                    head_size=self.head_size,
                    dropout=self.dropout,
                    is_decoder=self.is_decoder,
                )
                for _ in range(self.num_heads)
            ],
        )

        # if after concatenation the size of channels is bigger than embeddings size
        # projection will scaled it down
        self.projection = nn.Linear(self.head_size * self.num_heads, self.embeddings_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        """Apply multiple self-attention heads in parallel and concatenate the result.

        Parameters
        ----------
        x : Tensor
            vector representation of input token of size (batch, time-step, channels)

        Returns
        -------
        Tensor
            output vector of the same size as input
        """
        # concatenate over channel dimension
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.projection(out)
        return self.dropout(out)


# TODO: remove below line
# flake8: noqa
class CausalSelfAttention(nn.Module):
    def __init__(
        self,
        embeddings_size: int,
        context_size: int,
        head_size: None | int,
        num_heads: int,
        bias: bool,
        dropout: float,
    ) -> None:
        super().__init__()

        if not head_size:
            if embeddings_size % num_heads != 0:
                msg = "Embeddings size should be divisible by number of heads without residual, "
                f"but was provided: embeddings_size={embeddings_size}; num_heads={num_heads}"
                raise ValueError(msg)
            head_size = embeddings_size // num_heads
            # TODO: decide what to do with this
        logger.debug(
            "Embeddings_size {}, head_size {}, num_heads {}, head_size*num_heads={}".format(
                embeddings_size, head_size, num_heads, head_size * num_heads
            )
        )

        self.embeddings_size = embeddings_size
        self.context_size = context_size
        self.head_size = head_size
        self.num_heads = num_heads
        self.bias = bias
        self.dropout = dropout

        # key, query and value projections (hence `3 * embeddings_size`) for all heads in a single batch
        self.causal_self_attention = nn.Linear(embeddings_size, 3 * self.head_size * self.num_heads, bias=bias)
        # output projection
        # self.projection = nn.Linear(embeddings_size, embeddings_size, bias=bias)
        self.projection = nn.Linear(self.head_size * self.num_heads, embeddings_size, bias=bias)
        # regularization
        self.attention_dropout = nn.Dropout(self.dropout)
        self.projection_dropout = nn.Dropout(self.dropout)
        # triangular matrix for masking 'future' tokens
        self.register_buffer("tril", torch.tril(torch.ones(self.context_size, self.context_size)))

    def forward(self, x: Tensor) -> Tensor:
        # batch, sequence length, embedding size
        B, T, C = x.shape  # noqa: N806

        # query, key, value = self.causal_self_attention(x).split(
        #     self.embeddings_size,
        #     dim=-1,
        # )  # (B, T, C) -> (B, T, C * 3) -> (B, T, C)
        query, key, value = self.causal_self_attention(x).split(self.head_size * self.num_heads, dim=-1)

        # key = key.view(B, T, self.num_heads, C // self.num_heads).transpose(1, 2)  # (B, nh, T, hs)
        # query = query.view(B, T, self.num_heads, C // self.num_heads).transpose(1, 2)  # (B, nh, T, hs)
        # value = value.view(B, T, self.num_heads, C // self.num_heads).transpose(1, 2)  # (B, nh, T, hs)
        key = key.view(B, T, self.num_heads, self.head_size).transpose(1, 2)
        query = query.view(B, T, self.num_heads, self.head_size).transpose(1, 2)
        value = value.view(B, T, self.num_heads, self.head_size).transpose(1, 2)

        attention_scores = query @ key.mT  # (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        attention_scores *= key.shape[-1] ** -0.5  # (B, nh, T, T)
        attention_scores = attention_scores.masked_fill(self.tril[:T, :T] == 0, float("-inf"))  # (B, nh, T, T)
        attention_scores = F.softmax(attention_scores, dim=-1)  # (B, nh, T, T)
        attention_scores = self.attention_dropout(attention_scores)  # (B, nh, T, T)

        output = attention_scores @ value  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        # output = output.transpose(1, 2).contiguous().view(B, T, C)  # re-assemble all head outputs side by side
        output = (
            output.transpose(1, 2).contiguous().view(B, T, self.head_size * self.num_heads)
        )  # re-assemble all head outputs side by side
        # output projection
        output = self.projection(output)
        output = self.projection_dropout(output)

        return output
