# Derived from https://github.com/microsoft/LoRA
#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------

# flake8: noqa
# sourcery skip: default-mutable-arg, switch, remove-unnecessary-else

"""
            -----------------
            |       h       |
            -----------------
                    ^
                    |
                    +
                 /     \
    ----------------    -----------          Matrix initialization:
    |  pretrained  |     \       /           B = 0
    |   weights    |      \  B  /            A = N(0, sigma^2)
    |              |       -----
    |  W e R^(dxd) |       | r |             r - rank
    |              |       -----
    ----------------      /  A  \
            ^            /       \
             \          -----------
              \         ^
               \       /
            -----------------
            |       x       |
            -----------------

    With LoRA (low ranking adaptation) instead of learning weights of size d*d, we can freeze the
    pretrained weights and instead learn two matrices of size d*r and r*d: the number of parameters
    in this case will be reduced dramatically (depending on the rank of course) yet after multiplication
    of matrices d*r and r*d we will get a matrix d*d which we can sum with frozed pretrained weights and
    thus finetune model.
import math
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger

# import lit_llama.model as llama
from src.model.gpt_language_model import attention, transformer_block


class LoRALayer:
    def __init__(
        self,
        r: int,
        lora_alpha: int,
        lora_dropout: float,
        merge_weights: bool,
    ):
        self.r = r
        self.lora_alpha = lora_alpha
        # Optional dropout
        if lora_dropout > 0.0:
            self.lora_dropout = nn.Dropout(p=lora_dropout)
        else:
            self.lora_dropout = lambda x: x
        # Mark the weight as unmerged
        self.merged = False
        self.merge_weights = merge_weights


class MergedLinear(nn.Linear, LoRALayer):
    # LoRA implemented in a dense layer
    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        enable_lora: List[bool] = [False],
        # TODO: fan_in_fan_out is for initialization, right?
        fan_in_fan_out: bool = False,
        # TODO: how it affects everything?
        merge_weights: bool = True,
        **kwargs,
    ):
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout, merge_weights=merge_weights)
        # TODO: is it the right way to check?
        assert out_features % len(enable_lora) == 0, "The length of enable_lora must divide out_features"
        self.enable_lora = enable_lora
        self.fan_in_fan_out = fan_in_fan_out
        # Actual trainable parameters
        if r > 0 and any(enable_lora):
            # TODO: it feels like matrix multiplication should be A @ B, and here it looks like
            # B @ A, since B of size (..., r) and A - (r, ...)
            self.lora_A = nn.Parameter(self.weight.new_zeros((r * sum(enable_lora), in_features)))
            self.lora_B = nn.Parameter(
                self.weight.new_zeros((out_features // len(enable_lora) * sum(enable_lora), r))
            )  # weights for Conv1D with groups=sum(enable_lora)
            # TODO: scaling raises questions
            # We then scale ∆W x by α/r
            # where α is a constant in r. When optimizing with Adam, tuning α is roughly the same as tuning the learning
            # rate if we scale the initialization appropriately. As a result, we simply set α to the first r we try
            # and do not tune it. This scaling helps to reduce the need to retune hyperparameters when we vary r
            self.scaling = self.lora_alpha / self.r
            # Freezing the pre-trained weight matrix
            # TODO: feels weird: first freeze and then unfreeze
            # TODO: what does .weight means? There are weight for matrix A and matrix B, but what is it?
            self.weight.requires_grad = False
            # Compute the indices
            # TODO: indices I didn't understand
            self.lora_ind = self.weight.new_zeros((out_features,), dtype=torch.bool).view(len(enable_lora), -1)
            self.lora_ind[enable_lora, :] = True
            # .view(-1) is the same as .flatten()
            self.lora_ind = self.lora_ind.view(-1)
        self.reset_parameters()
        if fan_in_fan_out:
            self.weight.data = self.weight.data.T

    def reset_parameters(self):
        nn.Linear.reset_parameters(self)
        if hasattr(self, "lora_A"):
            # initialize A the same way as the default for nn.Linear and B to zero
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)

    def zero_pad(self, x):
        # TODO: don't know what it's doing ¯\_(ツ)_/¯
        result = x.new_zeros((*x.shape[:-1], self.out_features))
        result = result.view(-1, self.out_features)
        result[:, self.lora_ind] = x.reshape(-1, self.out_features // len(self.enable_lora) * sum(self.enable_lora))
        return result.view((*x.shape[:-1], self.out_features))

    def train(self, mode: bool = True):
        # TODO: don't know what it's doing ¯\_(ツ)_/¯
        def T(w):
            return w.T if self.fan_in_fan_out else w

        nn.Linear.train(self, mode)
        if self.merge_weights and self.merged:
            # Make sure that the weights are not merged
            if self.r > 0 and any(self.enable_lora):
                delta_w = F.conv1d(
                    self.lora_A.data.unsqueeze(0), self.lora_B.data.unsqueeze(-1), groups=sum(self.enable_lora)
                ).squeeze(0)
                self.weight.data -= self.zero_pad(T(delta_w * self.scaling))
            self.merged = False

    def eval(self):
        # TODO: don't know what it's doing ¯\_(ツ)_/¯
        def T(w):
            return w.T if self.fan_in_fan_out else w

        nn.Linear.eval(self)
        if self.merge_weights and not self.merged:
            # Merge the weights and mark it
            if self.r > 0 and any(self.enable_lora):
                delta_w = F.conv1d(
                    self.lora_A.data.unsqueeze(0), self.lora_B.data.unsqueeze(-1), groups=sum(self.enable_lora)
                ).squeeze(0)
                self.weight.data += self.zero_pad(T(delta_w * self.scaling))
            self.merged = True

    def forward(self, x: torch.Tensor):
        # TODO: don't know what it's doing ¯\_(ツ)_/¯
        def T(w):
            return w.T if self.fan_in_fan_out else w

        if self.merged:
            return F.linear(x, T(self.weight), bias=self.bias)
        else:
            result = F.linear(x, T(self.weight), bias=self.bias)
            if self.r > 0:
                after_A = F.linear(self.lora_dropout(x), self.lora_A)
                after_B = F.conv1d(
                    after_A.transpose(-2, -1), self.lora_B.unsqueeze(-1), groups=sum(self.enable_lora)
                ).transpose(-2, -1)
                result += self.zero_pad(after_B) * self.scaling
            return result


def mark_only_lora_as_trainable(model: nn.Module, bias: str = "none") -> None:
    for n, p in model.named_parameters():
        if "lora_" not in n:
            p.requires_grad = False
    if bias == "none":
        return
    elif bias == "all":
        for n, p in model.named_parameters():
            if "bias" in n:
                p.requires_grad = True
    elif bias == "lora_only":
        for m in model.modules():
            if isinstance(m, LoRALayer) and hasattr(m, "bias") and m.bias is not None:
                m.bias.requires_grad = True
    else:
        raise NotImplementedError


def lora_state_dict(model: nn.Module, bias: str = "none") -> Dict[str, torch.Tensor]:
    my_state_dict = model.state_dict()
    if bias == "none":
        return {k: my_state_dict[k] for k in my_state_dict if "lora_" in k}
    elif bias == "all":
        return {k: my_state_dict[k] for k in my_state_dict if "lora_" in k or "bias" in k}
    elif bias == "lora_only":
        to_return = {}
        for k in my_state_dict:
            if "lora_" in k:
                to_return[k] = my_state_dict[k]
                # TODO: check what is 'bias_name'
                bias_name = k.split("lora_")[0] + "bias"
                if bias_name in my_state_dict:
                    to_return[bias_name] = my_state_dict[bias_name]
        return to_return
    else:
        raise NotImplementedError


@dataclass
class LoRAConfig:
    r: float = 0.0
    alpha: float = 1.0
    dropout: float = 0.0


# class CausalSelfAttention(llama.CausalSelfAttention):
class CausalSelfAttention(attention.CausalSelfAttention):
    lora_config = None

    # def __init__(self, config: llama.LLaMAConfig) -> None:
    # def __init__(self, config: dict) -> None:
    def __init__(
        self,
        embeddings_size: int,
        context_size: int,
        head_size: Optional[int],
        num_heads: int,
        bias: bool,
        dropout: float,
        *,
        is_decoder: bool,
    ) -> None:
        logger.debug("Using Causal Self Attention with LoRA")
        # Skip the parent class __init__ altogether and replace it to avoid
        # useless allocations
        nn.Module.__init__(self)
        # assert config.n_embd % config.n_head == 0
        assert embeddings_size % num_heads == 0

        # key, query, value projections for all heads, but in a batch
        # TODO: what makes MergedLinear an attention layer
        #   apparently it's because of inheritance
        self.c_attn = MergedLinear(
            in_features=embeddings_size,
            out_features=3 * embeddings_size,
            r=self.lora_config.r,
            lora_alpha=self.lora_config.alpha,
            lora_dropout=self.lora_config.dropout,
            enable_lora=[True, False, True],
            fan_in_fan_out=False,
            merge_weights=True,
            bias=False,
        )
        # output projection
        self.projection = nn.Linear(embeddings_size, embeddings_size, bias=False)
        # TODO: is it indeed regularization section?
        # regularization
        self.num_heads = num_heads
        self.embeddings_size = embeddings_size
        self.context_size = context_size
        self.head_size = head_size
        self.bias = bias
        self.dropout = dropout
        self.is_decoder = is_decoder


@contextmanager
def lora(r, alpha, dropout, enabled: bool = True):
    """A context manager under which you can instantiate the model with LoRA."""
    if not enabled:
        yield
        return

    CausalSelfAttention.lora_config = LoRAConfig(r=r, alpha=alpha, dropout=dropout)

    # save original attention into a variable
    causal_self_attention = attention.CausalSelfAttention
    # replace original attention with LoRA variant
    attention.CausalSelfAttention = CausalSelfAttention
    transformer_block.CausalSelfAttention = CausalSelfAttention
    yield
    # reset original attention
    attention.CausalSelfAttention = causal_self_attention

    # causal_self_attention = llama.CausalSelfAttention
    # llama.CausalSelfAttention = CausalSelfAttention
    # yield
    # llama.CausalSelfAttention = causal_self_attention

    CausalSelfAttention.lora_config = None
