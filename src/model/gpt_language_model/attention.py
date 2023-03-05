import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class SelfAttentionHead(nn.Module):
    def __init__(
        self,
        embeddings_size: int,
        context_size: int,
        head_size: int,
        bias: bool,
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
        bias : bool
            whether to use bias or not: without bias might be a bit better and faster (but it's not for sure)
        dropout : float
            how many connection between tokens are dropped during each forward pass
        is_decoder : bool
            if it's a decoder masking of 'future' tokens will be applied
        """
        super().__init__()

        self.embeddings_size = embeddings_size
        self.context_size = context_size
        self.head_size = head_size
        self.bias = bias
        self.is_decoder = is_decoder

        # what don't need `bias` because we simply want to do matrix multiplications
        self.key_weights = nn.Linear(embeddings_size, head_size, bias=self.bias)
        self.query_weights = nn.Linear(embeddings_size, head_size, bias=self.bias)
        self.value_weights = nn.Linear(embeddings_size, head_size, bias=self.bias)

        # If you have parameters in your model, which should be saved and restored in the state_dict,
        # but not trained by the optimizer, you should register them as buffers.
        # Buffers won't be returned in model.parameters(), so that the optimizer won't have a change to update them.
        if self.is_decoder:
            self.register_buffer("tril", torch.tril(torch.ones(self.context_size, self.context_size)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor, kv_cache: None | Tensor) -> Tensor:
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
        kv_cache: None | Tensor
            key-value cache, but only if not None; if None - it means that it's disabled;
            contains cache for keys and value from all previous steps

        Returns
        -------
        Tensor
            output vector representation of x of size (batch, time-step, head_size)
        """
        # batch, time-step, channels (embeddings size)
        B, T, C = x.shape  # noqa: N806

        key = self.key_weights(x)  # (B, T, head_size)
        query = self.query_weights(x)  # (B, T, head_size)
        value = self.value_weights(x)  # (B, T, head_size)

        if kv_cache is not None:
            key_cached, value_cached = kv_cache.unbind(dim=0)  # (2, B, T, head_size) -> 2 * (B, T, head_size)
            key = torch.cat((key_cached, key), dim=-2)  # (B, cache + T, head_size)
            value = torch.cat((value_cached, value), dim=-2)  # (B, cache + T, head_size)

        # first to obtaining attention scores: dot product of key and query
        attention_scores = query @ key.mT  # (B, T, head_size) @ (B, head_size, T) -> (B, T, T)

        # In order to preserve 1 unit variance of the dot product of two vectors
        # we need to divide by square root of the features size (in our case - attention head size)
        # We need it to make sure that the values after softmax are well spread out, otherwise in worst
        # case scenario the values after the softmax will converge to one-hot encoding (like [0, 0, 1]) and
        # that will mean that the attention will be on a single (or couple of) tokens, and we want it to be
        # spread out (like [0.2, 0.1, 0.7])
        # we want to aggregate information not from a single node
        attention_scores /= math.sqrt(key.shape[-1])

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

        # return the weighted aggregation of the values and kv-cache if needed
        return (
            attention_scores @ value,  # (B, T, T) @ (B, T, head_size) -> (B, T, head_size)
            None if kv_cache is None else torch.stack((key, value)),  # None | (2, B, T, head_size)
        )


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        embeddings_size: int,
        context_size: int,
        head_size: None | int,
        num_heads: int,
        bias: bool,
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
        head_size : None | int
            the size of output of self-attention;
            if not provided `head_size` will be equal to `embeddings_size` // `num_heads`, so it should be divisible
            without remainder
        num_heads : int
            how many self-attention heads to use
        bias : bool
            whether to use bias or not: without bias might be a bit better and faster (but it's not for sure)
        dropout : float
            how many connection between tokens are dropped during each forward pass
        is_decoder : bool
            if it's a decoder masking of 'future' tokens will be applied

        Raises
        ------
        ValueError
            if `embeddings_size` cannot be divided by `num_heads` without remainder
        """
        super().__init__()

        if not head_size:
            if embeddings_size % num_heads != 0:
                msg = "Embeddings size should be divisible by number of heads without remainder, "
                f"but was provided: embeddings_size={embeddings_size}; num_heads={num_heads}"
                raise ValueError(msg)
            head_size = embeddings_size // num_heads

        self.embeddings_size = embeddings_size
        self.context_size = context_size
        self.head_size = head_size
        self.num_heads = num_heads
        self.bias = bias
        self.dropout = dropout
        self.is_decoder = is_decoder

        self.heads = nn.ModuleList(
            [
                SelfAttentionHead(
                    embeddings_size=self.embeddings_size,
                    context_size=self.context_size,
                    head_size=self.head_size,
                    bias=self.bias,
                    dropout=self.dropout,
                    is_decoder=self.is_decoder,
                )
                for _ in range(self.num_heads)
            ],
        )

        # if after concatenation the size of channels is bigger than embeddings size
        # projection will scaled it down
        self.projection = nn.Linear(self.head_size * self.num_heads, self.embeddings_size, bias=self.bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor, kv_cache: None | Tensor) -> Tensor:
        """Apply multiple self-attention heads in parallel and concatenate the result.

        Parameters
        ----------
        x : Tensor
            vector representation of input token of size (batch, time-step, channels)
        kv_cache: None | Tensor
            key-value cache, but only if not None; if None - it means that it's disabled;
            contains cache for keys and value from all previous steps

        Raises
        ------
        ValueError
            if new key-value cache is mixed with None and non-None values

        Returns
        -------
        Tensor
            output vector of the same size as input
        """
        # prepare kv-cache for each head
        if kv_cache is None:
            kv_cache = [None] * self.num_heads  # nh * None
        # if kv-cache is not an empty tensor
        elif kv_cache.numel():
            kv_cache = kv_cache.unbind(2)  # (2, B, num_heads, T, head_size) -> num_heads * (2, B, T, head_size)
        else:
            kv_cache = [kv_cache.detach().clone() for _ in range(self.num_heads)]  # num_heads * empty_tensor

        # apply head and corresponding kv-cache on an input
        outs = [head(x, kv_head) for head, kv_head in zip(self.heads, kv_cache)]
        outputs, kv_cache = zip(*outs)

        output = torch.cat(outputs, dim=-1)  # num_heads * (B, T, head_size) -> (B, T, num_heads * head_size)
        output = self.dropout(self.projection(output))  # (B, T, num_heads * head_size)

        # has to be all None or all non-None
        if any(x is None for x in kv_cache) and any(x is not None for x in kv_cache):
            raise ValueError("Mixed list of None and non-None values in kv-cache.")

        if all(x is not None for x in kv_cache):
            kv_cache = torch.stack(
                kv_cache,
                dim=-2,
            )  # num_heads * (2, B, T, head_size) -> (2, B, T, num_heads, head_size)
            kv_cache = kv_cache.transpose(2, 3)  # (2, B, num_heads, T, head_size)

        return (
            output,  # (B, T, num_heads * head_size)
            kv_cache,  # num_heads * None | (2, B, num_heads, T, head_size)
        )


class CausalSelfAttention(nn.Module):
    def __init__(
        self,
        embeddings_size: int,
        context_size: int,
        head_size: None | int,
        num_heads: int,
        bias: bool,
        dropout: float,
        *,
        is_decoder: bool,
    ) -> None:
        """Do the same as multi-head attention but with a single matrix multiplication.

        Instead of creating multiple heads and concatenating the result (in addition to creating separate matrices for
        query, key and value for each head) we can do this in a single pass with a single weight matrix.

        Parameters
        ----------
        embeddings_size : int
            size of the embeddings - the size of input of self-attention
        context_size : int
            the number of tokens that will be used during calculation attention map and
            weighted averaging of value of each token
        head_size : None | int
            the size of output of self-attention;
            if not provided `head_size` will be equal to `embeddings_size` // `num_heads`, so it should be divisible
            without remainder
        num_heads : int
            how many self-attention heads to use
        bias : bool
            whether to use bias or not: without bias might be a bit better and faster (but it's not for sure)
        dropout : float
            how many connection between tokens are dropped during each forward pass
        is_decoder : bool
            if it's a decoder masking of 'future' tokens will be applied

        Raises
        ------
        ValueError
            if `embeddings_size` cannot be divided by `num_heads` without remainder
        """
        super().__init__()

        if not head_size:
            if embeddings_size % num_heads != 0:
                msg = "Embeddings size should be divisible by the number of heads without a residual, "
                f"but was provided: embeddings_size={embeddings_size}; num_heads={num_heads}"
                raise ValueError(msg)
            head_size = embeddings_size // num_heads

        self.embeddings_size = embeddings_size
        self.context_size = context_size
        self.head_size = head_size
        self.num_heads = num_heads
        self.bias = bias
        self.dropout = dropout
        self.is_decoder = is_decoder

        # key, query and value projections (hence `3 * ...`) for all heads in a single batch
        self.causal_self_attention = nn.Linear(embeddings_size, 3 * self.head_size * self.num_heads, bias=self.bias)
        # output projection
        self.projection = nn.Linear(self.head_size * self.num_heads, embeddings_size, bias=self.bias)
        # regularization
        self.attention_dropout = nn.Dropout(self.dropout)
        self.projection_dropout = nn.Dropout(self.dropout)
        # triangular matrix for masking 'future' tokens
        if self.is_decoder:
            self.register_buffer("tril", torch.tril(torch.ones(self.context_size, self.context_size)))

    def forward(self, x: Tensor, kv_cache: None | Tensor) -> Tensor:
        """Do multi-head attention in a single pass.

        Multiply by weight matrix -> split the result into query, key and value -> reshape each one of them
        into shape (batch, num_heads, time-steps, head_size). The rest is similar to single self-attention head
        forward pass.

        Parameters
        ----------
        x : Tensor
            input tensor of shape (batch, time-step, embedding size)
        kv_cache: None | Tensor
            key-value cache, but only if not None; if None - it means that it's disabled;
            contains cache for keys and value from all previous steps

        Returns
        -------
        Tensor
            output tensor of the same shape as input: (batch, time-step, embedding size)
        """
        # notation:
        # - B  | batch
        # - T  | time-step (sequence length)
        # - C  | embeddings size
        # - hs | head size
        # - nh | number of heads

        B, T, C = x.shape  # noqa: N806

        # single pass for query, key and value; that's why we need to split into 3 parts
        query, key, value = self.causal_self_attention(x).split(
            self.head_size * self.num_heads,
            dim=-1,
        )  # (B, T, C) -> (B, T, 3 * hs * nh) -> (B, T, hs * nh)

        # transform (B, T, nh * hs) -> (B, nh, T, hs) so it's similar to multi-head attention
        key = key.view(B, T, self.num_heads, self.head_size).transpose(1, 2)  # (B, nh, T, hs)
        query = query.view(B, T, self.num_heads, self.head_size).transpose(1, 2)  # (B, nh, T, hs)
        value = value.view(B, T, self.num_heads, self.head_size).transpose(1, 2)  # (B, nh, T, hs)

        if kv_cache is not None:
            key_cached, value_cached = kv_cache.unbind(dim=0)  # (2, B, T, head_size) -> 2 * (B, T, head_size)
            key = torch.cat((key_cached, key), dim=-2)  # (B, cache + T, head_size)
            value = torch.cat((value_cached, value), dim=-2)  # (B, cache + T, head_size)

        # to obtain attention scores first do dot product of query and key
        attention_scores = query @ key.mT  # (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)

        # In order to preserve 1 unit variance of the dot product of two vectors
        # we need to divide by square root of the features size (in our case - attention head size)
        # We need it to make sure that the values after softmax are well spread out, otherwise in worst
        # case scenario the values after the softmax will converge to one-hot encoding (like [0, 0, 1]) and
        # that will mean that the attention will be on a single (or couple of) tokens, and we want it to be
        # spread out (like [0.2, 0.1, 0.7])
        # we want to aggregate information not from a single node
        attention_scores /= math.sqrt(key.shape[-1])  # (B, nh, T, T)

        # if it's a decoder we need to mask 'future' tokens with '-inf' value
        if self.is_decoder:
            # [0.9, -0.6, 0.3] -> [0.9, -inf, -inf]
            # [0.1, 0.5, -0.1] -> [0.1, 0.5, -inf]
            # [0.1, 0.2, 0.3]  -> [0.1, 0.2, 0.3]
            # and after softmax `-inf` becomes 0
            # this doesn't allow current token communicate with future ones
            attention_scores = attention_scores.masked_fill(self.tril[:T, :T] == 0, float("-inf"))  # (B, nh, T, T)

        # since we want to do weighted averaging we need to transform attention scores into range [0, 1]
        # and sum of all scores should be equal to 1; softmax is a good tool for it
        attention_scores = F.softmax(attention_scores, dim=-1)  # (B, nh, T, T)

        # randomly prevent some nodes from communicating, some of theme randomly are set to zero
        # helps prevent overfitting
        attention_scores = self.attention_dropout(attention_scores)  # (B, nh, T, T)

        # perform the weighted aggregation of the values
        output = attention_scores @ value  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        # re-assemble all head outputs side by side
        output = output.transpose(1, 2).reshape(B, T, self.head_size * self.num_heads)  # (B, T, hs * nh)
        # output projection
        output = self.projection(output)  # (B, T, C)
        return (
            self.projection_dropout(output),  # (B, T, C)
            None if kv_cache is None else torch.stack((key, value)),  # None | # (2, B, nh, T, hs)
        )
