import torch
import torch.nn as nn
from torch.nn import functional as F


class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size) -> None:
        super().__init__()
        # each token directly reads off the logits for the next token from the lookup table
        # in other words for each token "feature" is the "frequency" for each next word
        # that why it has square size
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None) -> tuple[torch.Tensor, None | torch.Tensor]:

        # idx and target are both (B, T) tensor of integers
        # where B - batch size, T - num steps (context size)
        logits = self.token_embedding_table(idx)  # (B, T, C), where C - number of "features"

        # if not targets:
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            # TODO: remember why do we do that
            # something related to order of channels
            loss = F.cross_entropy(
                logits.view(B * T, C),
                targets.view(B * T),
            )

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # get the predictions
            logits, loss = self(idx)  # (B, T, C)
            # focus only on the last time step
            logits = logits[:, -1, :]  # becomes (B, C)
            # apply softmax on the predictions to get probabilities
            # TODO: why we specify 'dim'
            probs = F.softmax(logits, dim=-1)  # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T + 1)
        return idx
