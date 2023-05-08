# Derived from https://github.com/microsoft/LoRA
#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------

# flake8: noqa
# sourcery skip: default-mutable-arg, switch, remove-unnecessary-else

"""

    Low Ranking Adaptation for LLMs scheme

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
    in this case will be reduced drastically (depending on the rank of course) yet after multiplication
    of matrices d*r and r*d we will get a matrix d*d which we can sum with frozen pretrained weights and
    thus finetune model.

    The goal of this approach is to move weight updates into a separete matrix which is decomposed with
    two matrices of a lower rank.
"""

import math
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger

from src.model.gpt_language_model import attention, transformer_block
from src.utils.error import log_error


class LoRALayer:
    def __init__(
        self,
        r: int,
        lora_alpha: int,
        lora_dropout: float,
        merge_weights: bool,
    ):
        """Base class to store LoRA specific attributes.

        Parameters
        ----------
        r : int
            rank of the weight update matrices. To make sense of using LoRA the rank should be smaller than the rank of
            the weights of the model.  The rank can be as low as 1: https://arxiv.org/pdf/2106.09685.pdf (section 7.2)
        lora_alpha : int
            alpha is needed for scaling updates as alpha/r
            "This scaling helps to reduce the need to retune hyperparameters when we vary r"
            https://arxiv.org/pdf/2106.09685.pdf (section 4.1)
        lora_dropout : float
            dropout that is applied on the input in the LoRA branch (before multiplying by matrix A)
        merge_weights : bool
            whether we want to merge weights and LoRA weight updates. This is useful if one wants to use finetuned model
            as a standalone one (without storing LoRA weight separately) plus it helps to reduce overhead.
        """
        self.r = r
        self.lora_alpha = lora_alpha
        # Optional dropout
        if lora_dropout > 0.0:
            self.lora_dropout = nn.Dropout(p=lora_dropout)
        else:
            self.lora_dropout = lambda x: x
        # Mark the weight as unmerged
        self.merged = False  # stores status if the weights are already merged
        self.merge_weights = merge_weights


class MergedLinear(nn.Linear, LoRALayer):
    # LoRA implemented in a dense layer
    def __init__(
        self,
        # this part is for pretrained weights
        in_features: int,
        out_features: int,
        # -----------
        # this part is for LoRA
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        # bool value for query, key and value matrices if we want to apply LoRA to not all of them
        enable_lora: List[bool] = [False],
        # # NOTE: fan_in_fan_out (`bool`): Set this to True if the layer to replace stores weight like (fan_in, fan_out).
        # For example, gpt-2 uses `Conv1D` which stores weights like (fan_in, fan_out) and hence this should be set to `True`
        # https://github.com/huggingface/peft/blob/main/src/peft/tuners/lora.py#LL53C9-L53C112
        fan_in_fan_out: bool = False,  # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
        # NOTE: tells to merge weights: pretrained weights with LoRA's A and B matrices
        merge_weights: bool = True,
        **kwargs,
    ):
        # NOTE: i assume that self.weight comes from nn.Linear parent class
        #   and it stores pretrained weights that we are not planning to update
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout, merge_weights=merge_weights)
        # TODO: is it the right way to check?
        # NOTE: seems like it is
        assert out_features % len(enable_lora) == 0, "The length of enable_lora must divide out_features"
        self.enable_lora = enable_lora
        self.fan_in_fan_out = fan_in_fan_out
        # Actual trainable parameters

        # in_features: 128 (embeddings_size)
        # out_features: 384 (3 * embeddings_size)
        if r > 0 and any(enable_lora):
            # TODO: it feels like matrix multiplication should be A @ B, and here it looks like
            # B @ A, since B of size (..., r) and A - (r, ...)
            # NOTE: perhaps it's somehow connected to the fact that we use Conv1D instead of Linear layers
            self.lora_A = nn.Parameter(self.weight.new_zeros((r * sum(enable_lora), in_features)))  # (4, 128)
            self.lora_B = nn.Parameter(
                self.weight.new_zeros((out_features // len(enable_lora) * sum(enable_lora), r))  # (256, 2)
            )  # weights for Conv1D with groups=sum(enable_lora)

            # We then scale ∆W x by α/r
            # where α is a constant in r. When optimizing with Adam, tuning α is roughly the same as tuning the learning
            # rate if we scale the initialization appropriately. As a result, we simply set α to the first r we try
            # and do not tune it. This scaling helps to reduce the need to retune hyperparameters when we vary r
            # https://arxiv.org/pdf/2106.09685.pdf, page 4

            # This balances the pretrained model’s knowledge and the new task-specific adaptation
            # https://lightning.ai/pages/community/tutorial/lora-llm/

            # So, set alpha to 1.0 to fully add LoRA. If the LoRA seems to have too much effect (i.e., overfitted), set
            # alpha to lower value. If the LoRA seems to have too little effect, set alpha to higher than 1.0. You can
            # tune these values to your needs. This value can be even slightly greater than 1.0!
            # https://github.com/cloneofsimo/lora
            self.scaling = self.lora_alpha / self.r

            # Freezing the pre-trained weight matrix
            # This class will stores three matrices:
            # - .weight: stores pretrained weights
            # - .lora_A and .lora_B: are used for LoRA and stores weight updates
            self.weight.requires_grad = False  # (384, 128)

            # Compute the indices
            # NOTE: apparently we need indices when we want to apply lora not for all three matrices: q, k and v

            # Basically here we create a matrix with len of 'out_features', then represent it as 2D matrix
            # (len(enable_lora), -1), set all values in the corresponding row as True and after it just
            # flatten 2D matrix into 1D
            # TODO: feels weird, there should be a better way to do it
            self.lora_ind = self.weight.new_zeros((out_features,), dtype=torch.bool).view(
                len(enable_lora), -1
            )  # (3, 128)
            self.lora_ind[enable_lora, :] = True  # (3, 128)
            # .view(-1) is the same as .flatten()
            self.lora_ind = self.lora_ind.view(-1)  # (384,)
        self.reset_parameters()
        if fan_in_fan_out:
            self.weight.data = self.weight.data.T

    def reset_parameters(self):
        # Apparently this method is used when we want to reset all weights, even including
        # pretrained ones, because `.reset_parameters` method will reset all the weights
        nn.Linear.reset_parameters(self)
        if hasattr(self, "lora_A"):
            # initialize A the same way as the default for nn.Linear and B to zero
            # TODO: find out what is 'a' and why it's equal to math.sqrt(5)
            # NOTE: https://github.com/pytorch/pytorch/issues/15314
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)

    def zero_pad(self, x):  # (64, 64, 256)
        # NOTE: presumably it's needed when not for all matrices (out of q, k, v) LoRA is applied
        # For example if we don't want to fine-tune weight for key matrix, so the LoRA updates for
        # key matrix section will be all filled with zeros
        # [[ΔW, ΔW, ΔW, 0, 0, 0, ΔW, ΔW, ΔW]
        #  [ΔW, ΔW, ΔW, 0, 0, 0, ΔW, ΔW, ΔW]
        #  [ΔW, ΔW, ΔW, 0, 0, 0, ΔW, ΔW, ΔW]]
        result = x.new_zeros((*x.shape[:-1], self.out_features))  # (64, 64, 384)
        result = result.view(-1, self.out_features)  # (4096, 384)
        result[:, self.lora_ind] = x.reshape(
            -1, self.out_features // len(self.enable_lora) * sum(self.enable_lora)
        )  # (4096, 256)
        return result.view((*x.shape[:-1], self.out_features))  # (64, 64, 384)

    def train(self, mode: bool = True):
        # NOTE: looks like this method does two things:
        # 1. sets the class into train mode (should affect Dropout)
        # 2. subtract LoRA updates from main weight matrix, so we can train A and B matrices separately

        # NOTE: feels weird to transpose it each time
        def T(w):
            return w.T if self.fan_in_fan_out else w

        # if train(True) -> unmerge unless we already have them unmerged
        # if train(False) -> merge unless we already have them merged
        # in train mode we will do anything if weights are already merged
        # in eval mode  we will do anything if weights are not merged
        should = self.merged if mode else not self.merged

        # TODO: does it affects pretrained weights?
        # TODO: if it's called from nn.Linear will it affect nn.Dropout?
        # NOTE: despite it being called from nn.Linear this method will put
        #   all layers into train mode, including nn.Dropout
        #   except parameters (such as self.lora_A, self.lora_B)
        nn.Linear.train(self, mode)
        # if we want to merge weight and they are already merged
        if self.merge_weights and should:
            # Make sure that the weights are not merged
            if self.r > 0 and any(self.enable_lora):
                # lora_A.data (4, 128) -> (1, 4, 128)
                # lora_B.data (256, 2) -> (256, 2, 1)
                delta_w = F.conv1d(
                    self.lora_A.data.unsqueeze(0),
                    self.lora_B.data.unsqueeze(-1),
                    groups=sum(self.enable_lora),  # (1, 4, 128) @ (256, 2, 1) -> (1, 256, 128)
                ).squeeze(
                    0
                )  # (1, 256, 128) -> (256, 128)
                # TODO: shape mismatch: value tensor of shape [128, 256] cannot be broadcast to indexing result of shape [256, 256]
                # -1: W = W - delta_W (unmerge), +1: W = W + delta_W (merge)
                sign = -1 if mode else 1
                self.weight.data += sign * self.zero_pad(T(delta_w * self.scaling))  # (256, 128) -> (384, 128)
            self.merged = not mode

    # def eval(self):
    #     # NOTE: looks like this method does two things:
    #     # 1. sets the class into eval mode (should affect Dropout)
    #     # 2. adds LoRA updates into the main weight matrix, so it lets us to use weights
    #     #    as a standalone model and should reduce overhead during inference

    #     def T(w):
    #         return w.T if self.fan_in_fan_out else w

    #     nn.Linear.eval(self)
    #     if self.merge_weights and not self.merged:
    #         # Merge the weights and mark it
    #         if self.r > 0 and any(self.enable_lora):
    #             delta_w = F.conv1d(
    #                 self.lora_A.data.unsqueeze(0), self.lora_B.data.unsqueeze(-1), groups=sum(self.enable_lora)
    #             ).squeeze(0)
    #             self.weight.data += self.zero_pad(T(delta_w * self.scaling))
    #         self.merged = True

    def forward(self, x: torch.Tensor):
        # NOTE: if weights are merged then we can simply do matmul between input tensor x and weight matrix
        #       if not merged, we need to do matmul and then apply LoRA weight update
        def T(w):
            return w.T if self.fan_in_fan_out else w

        # batch, context, embedding_size
        # B, T, C = x.shape # (64, 64, 128)

        # the logic here is that the weight are merged only during training
        # so if they are merged we don't need to do anything with LoRA's A and B matrices
        # since forward method is called during inference
        if self.merged:
            # x: (64, 64, 128)
            # self.weight: (384, 128)
            # return F.linear(x, T(self.weight), bias=self.bias)
            result = F.linear(x, T(self.weight), bias=self.bias)
            return result
        else:
            # if it's training (or more specifically - finetuning)
            # NOTE: `F.linear` automatically transposes the second argument (T(self.weight) in our case)
            result = F.linear(x, T(self.weight), bias=self.bias)  # (64, 64, 128) @ (384, 128) -> (64, 64, 384)
            if self.r > 0:
                # TODO; why for matrix A we use F.linear, while for B we use conv1D between matrices A and B?
                #       is it because we can use groups?
                #       also it might be faster to use Conv1D on a GPU
                #       https://discuss.pytorch.org/t/how-is-a-conv1d-with-groups-1-different-from-a-linear-layer/100505/2
                #       but the link show example with groups=1
                # TODO: check what 'groups' gives us
                after_A = F.linear(self.lora_dropout(x), self.lora_A)  # (64, 64, 128) @ (4, 128) -> (64, 64, 4)
                # NOTE: input – input tensor of shape (minibatch,in_channels,iW)
                #       weight – filters of shape (out_channels,in_channels/groups,kW)
                #       groups – split input into groups, in_channels should be divisible by the number of groups. Default: 1
                # perhaps kW - kernel width, iW - sequence width/length?
                after_B = F.conv1d(
                    after_A.transpose(-2, -1),
                    self.lora_B.unsqueeze(-1),
                    groups=sum(self.enable_lora),  # (64, 4, 64) @ (256, 2, 1) -> (64, 256, 64)
                ).transpose(
                    -2, -1
                )  # (64, 256, 64) -> (64, 64, 256)

                # (64, 64, 256) after zero_pad (64, 64, 384)
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
        # TODO: use __init__ from parent class
        # Skip the parent class __init__ altogether and replace it to avoid
        # useless allocations
        nn.Module.__init__(self)
        # assert config.n_embd % config.n_head == 0
        assert embeddings_size % num_heads == 0

        if not head_size:
            if embeddings_size % num_heads != 0:
                log_error(
                    "Embeddings size should be divisible by the number of heads without a residual, "
                    f"but was provided: embeddings_size={embeddings_size}; num_heads={num_heads}",
                )
            head_size = embeddings_size // num_heads

        # key, query, value projections for all heads, but in a batch
        self.causal_self_attention = MergedLinear(
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
        if self.is_decoder:
            self.register_buffer("tril", torch.tril(torch.ones(self.context_size, self.context_size)))
        self.attention_dropout = nn.Dropout(self.dropout)
        self.projection_dropout = nn.Dropout(self.dropout)


@contextmanager
def lora(r, alpha, dropout, enabled: bool = True):
    # TODO: make it pretty
    """A context manager under which you can instantiate the model with LoRA."""
    if not enabled:
        yield
        return

    CausalSelfAttention.lora_config = LoRAConfig(r=r, alpha=alpha, dropout=dropout)

    # save original attention into a variable
    causal_self_attention = attention.CausalSelfAttention
    block_causal_self_attention = transformer_block.CausalSelfAttention
    # replace original attention with LoRA variant
    attention.CausalSelfAttention = CausalSelfAttention
    transformer_block.CausalSelfAttention = CausalSelfAttention
    yield
    # reset original attention
    attention.CausalSelfAttention = causal_self_attention
    transformer_block.CausalSelfAttention = block_causal_self_attention

    # causal_self_attention = llama.CausalSelfAttention
    # llama.CausalSelfAttention = CausalSelfAttention
    # yield
    # llama.CausalSelfAttention = causal_self_attention

    CausalSelfAttention.lora_config = None


if __name__ == "__main__":
    model = MergedLinear(128, 384, 2, 1, 0.1, [True, False, True])
    model.eval()
    # model.train(False)
