import torch
import torch.nn.functional as F
from torch import Tensor, nn


class Head(nn.Module):
    def __init__(self, n_embed: int, head_size: int, block_size: int, dropout: float) -> None:
        super().__init__()

        self.n_embed = n_embed
        self.head_size = head_size
        self.block_size = block_size

        self.key = nn.Linear(in_features=n_embed, out_features=head_size, bias=False)
        self.query = nn.Linear(in_features=n_embed, out_features=head_size, bias=False)
        self.value = nn.Linear(in_features=n_embed, out_features=head_size, bias=False)
        # If you have parameters in your model, which should be saved and restored in the state_dict,
        # but not trained by the optimizer, you should register them as buffers.
        # Buffers won’t be returned in model.parameters(), so that the optimizer won’t have a change to update them.
        self.register_buffer("tril", torch.tril(torch.ones(self.block_size, self.block_size)))

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: Tensor) -> Tensor:
        b, t, c = x

        k: Tensor = self.key(x)  # (B, T, C)
        q: Tensor = self.query(x)  # (B, T, C)

        # compute attention scores ("affinities")
        wei: Tensor = q @ k.transpose(-2, -1)  # (B, T, C) @ (B, C, T) -> (B, T, T)
        # TODO: write down why to do that
        wei *= k.shape[-1] ** -0.5
        wei = wei.masked_fill(self.tril[:t, :t] == 0, float("-inf"))  # (B, T, T)
        wei = F.softmax(wei, dim=-1)  # (B, T, T)
        wei = self.dropout(wei)

        # perform the weighted aggregation of the values
        v: Tensor = self.value(x)  # (B, T, C)
        out = wei @ v  # (B, T, T) @ (B, T, C) -> (B, T, C)
        return out


class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads: int, head_size: int, n_embed: int, dropout: float, block_size: int) -> None:
        super().__init__()

        self.num_heads = num_heads
        self.head_size = head_size
        self.n_embed = n_embed
        self.block_size = block_size

        self.heads = nn.ModuleList(
            [
                Head(
                    head_size=self.head_size,
                    block_size=self.block_size,
                    dropout=self.dropout,
                    n_embed=self.n_embed,
                )
                for _ in self.num_heads
            ]
        )

        self.projection = nn.Linear(self.n_embed, self.n_embed)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: Tensor) -> Tensor:
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.projection(out)
        return self.dropout(out)
