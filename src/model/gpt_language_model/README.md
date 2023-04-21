# Notes about Transformer architecture

![transformer architecture](../../../assets/transformer/transformer_architecture.png)

pic 1: Transformer architecture[^1] (encoder on the left and decoder is on the right side).

In this repository the focus is on decoder part of transformer architecture as the intent is to generate new text alike text in tiny-shakespeare dataset.

Decoder consists of transformer blocks and each transformer block consists of two distinct layers (actually three, but for now we will not discuss encoder-decoder attention as we don't use encoder):

1. Self-attention layer.
2. Feed-forward layer.

![transformer block](../../../assets/transformer/transformer_block.png)

pic 2: Transformer block[^1].

Now let's talk about each layer of transformer block in more details.

## Self-attention

Self-attention layer is used to have such embeddings for each token that the embedding vector contains not only information about the token itself but also about token\tokens in which it is interested the most (highest attention).

It is done via transforming each token embedding into key, query and value vector through dot products of vector of token x and weight matrices K, Q an V.

Intuitively speaking:

- query: what the token is interested in *(a vector you want to calculate attention for)*
- key: what the token represents *(a vector you want to calculate attention against)*
- value: if someone is interested in the token, that's the value will be returned

As we obtain key and value vectors from token x itself (and not from external source) means that we perform self-attention.

Then by multiplying key and query for each token and passing it through softmax operation, we obtain attention map, which we can use in order to do weight averaging of values. If it doesn't make sense to you please read this awesome article[^1].

More notes about attention from Andrej Karpathy's notebook[^2]:

- Attention is a **communication mechanism**. Can be seen as nodes in a directed graph looking at each other and aggregating information with a weighted sum from all nodes that point to them, with data-dependent weights.
- There is no notion of space. Attention simply acts over a set of vectors. This is why we need to positionally encode tokens.
- Each example across batch dimension is of course processed completely independently and never "talk" to each other
- In an "encoder" attention block just delete the single line that does masking with `tril`, allowing all tokens to communicate. This block here is called a "decoder" attention block because it has triangular masking, and is usually used in autoregressive settings, like language modeling.
- "self-attention" just means that the keys and values are produced from the same source as queries. In "cross-attention", the queries still get produced from x, but the keys and values come from some other, external source (e.g. an encoder module)
- "Scaled" attention additional divides `wei` by 1/sqrt(head_size). This makes it so when input Q,K are unit variance, wei will be unit variance too and Softmax will stay diffuse and not saturate too much.

## Feed-forward

Then we have simple fully connected layer/layers. Why do we need this?

If there is no feed-forward layer, self-attention is simply a process of re-averaging of value vectors. In order to add element-wise non-linearity transformation of incoming vectors we add feed-forward part.

You can think about it in this way:

- attention step is for communication between tokens
- feed-forward is for processing this information (of how tokens are related to each other via attention)

The same feed-forward layer is applied on each self-attention output independently.

## AddNorm

Also one can notice additional step in each transformer block: addition (residual connection) and normalization operations. Both are used in order to be able to effectively build deep learning models (with many layers).

## Positional embeddings

Since attention doesn't have any notion of space and we want to preserve this information positional embeddings are used. They are simply added to word embeddings and contains information of relative position of each token. Positional embeddings are optimized during training the model.

## The whole model

In order to build decoder one needs to have:

1. Word embeddings layer.
2. Positional embeddings layer.
3. Multiple transformer blocks (self-attention, optionally encoder-decoder attention, feed-forward layer and do not forget about residual connections and normalization). The number of blocks is a hyperparameter.
4. Final head fully-connected layer to transform final token embeddings into predictions.

[^1]: [Illustrated transformer](https://jalammar.github.io/illustrated-transformer/)
[^2]:[Andrej Karpaty's nanoGPT Google Colab](<https://colab.research.google.com/drive/1JMLa53HDuA-i7ZBmqV7ZnA3c_fvtXnx-?usp=sharing#scrollTo=h5hjCcLDr2WC>)
