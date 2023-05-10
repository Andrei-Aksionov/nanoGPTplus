# Derived from https://github.com/microsoft/LoRA
# Derived from https://github.dev/Lightning-AI/lit-llama
#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------

r"""
    Low Ranking Adaptation for LLMs scheme.

            -----------------
            |       h       |
            -----------------
                    ^
                    |
                    +
                 /     \
    ----------------    -----------          Matrix initialization:
    |  pretrained  |     \   B    /          B = 0
    |   weights    |      \ r*d  /           A = N(0, sigma^2)
    |              |       -----
    |  W e R^(d*d) |       | r |             r - rank
    |              |       -----
    ----------------      /  A  \
            ^            /  d*r  \
             \          -----------
              \         ^
               \       /
            -----------------
            |       x       |
            -----------------

    With LoRA (Low Ranking Adaptation) instead of learning weights of size d*d, we can freeze the pretrained weights and
    instead learn two matrices of size d*r and r*d (they will store weight updates for the pretrained weights): the
    number of parameters in this case will be reduced drastically (depending on the rank of course) yet after
    multiplication of matrices d*r and r*d we will get a matrix d*d which we can sum with frozen pretrained weights and
    thus finetune model.

    The goal of this approach is to move weight updates into a separete matrix which is decomposed with
    two matrices of a lower rank.
"""  # noqa: D208

import math
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F
from loguru import logger
from torch import nn

from src.model.gpt_language_model import attention, transformer_block

# TODO: refactor original code in a way that it helps to get rid of noqa statements


class LoRALayer:
    def __init__(
        self,
        r: int,
        lora_alpha: int,
        lora_dropout: float,
        merge_weights: bool,
    ) -> None:
        """Store LoRA specific attributes in a class.

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
            whether we want to merge pretrained weights and LoRA weight updates. This is useful if one wants to use
            finetuned model as a standalone one (without storing LoRA weight separately) plus it helps to reduce
            overhead.
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
    def __init__(
        self,
        # ↓ this part is for pretrained weights
        in_features: int,
        out_features: int,
        # ↓ the remaining part is for LoRA
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        enable_lora: List[bool] = (False, False, False),
        fan_in_fan_out: bool = False,
        merge_weights: bool = True,
        **kwargs,
    ) -> None:
        """LoRA wrapper around linear class that is used for calculation of q, k and v matrices.

        This class has three weight matrices:
            1. Pretrained weights are stored as `self.weight` (because of the nn.Linear inheritance)
            2. LoRA A matrix as `self.lora_A`
            3. LoRA B matrix as `self.lora_B`
        Only LoRA's A and B matrices are updated, pretrained weights stay frozen.

        Parameters
        ----------
        in_features : int
            number of input features of the pretrained weights
        out_features : int
            number of output features of the pretrained weights
        r : int, optional
            rank of the weight update matrices. To make sense of using LoRA the rank should be smaller than the rank of
            the weights of the model.  The rank can be as low as 1: https://arxiv.org/pdf/2106.09685.pdf (section 7.2)
        lora_alpha : int
            alpha is needed for scaling updates as alpha/r
            "This scaling helps to reduce the need to retune hyperparameters when we vary r"
            https://arxiv.org/pdf/2106.09685.pdf (section 4.1)
        lora_dropout : float
            dropout that is applied on the input in the LoRA branch (before multiplying by matrix A)
        enable_lora : List[bool], optional
            MergeLinear class is for attention mechanism where qkv are calculated with a single weight matrix. If we
            don't want to apply LoRA for all three (query, key and value) we can set it as False. For example if we want
            to apply LoRA only to `query` and `value` but keep `key` without weight updates we should pass `[True,
            False, True]`
        fan_in_fan_out : bool, optional
            set this to True if the layer to replace stores weight like (fan_in, fan_out).  For example, gpt-2 uses
            `Conv1D` which stores weights like (fan_in, fan_out) and hence this should be set to `True`
            https://github.com/huggingface/peft/blob/main/src/peft/tuners/lora.py#LL53C9-L53C112
        merge_weights : bool
            whether we want to merge pretrained weights and LoRA weight updates. This is useful if one wants to use
            finetuned model as a standalone one (without storing LoRA weight separately) plus it helps to reduce
            overhead.
        """
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout, merge_weights=merge_weights)
        assert out_features % len(enable_lora) == 0, "The length of enable_lora must divide out_features"  # noqa: S101
        self.enable_lora = enable_lora
        self.fan_in_fan_out = fan_in_fan_out

        # Actual trainable parameters
        # To better understand initialization let's imagine that we have such parameters:
        # ⚬ in_features: 128 (embeddings_size)
        # ⚬ out_features: 384 (3 * embedding_size)
        # ⚬ r: 2
        # ⚬ enable_lora: [True, False, True]
        if r > 0 and any(enable_lora):
            self.lora_A = nn.Parameter(self.weight.new_zeros((r * sum(enable_lora), in_features)))  # (4, 128)
            self.lora_B = nn.Parameter(
                self.weight.new_zeros((out_features // len(enable_lora) * sum(enable_lora), r)),  # (256, 2)
            )  # weights for Conv1D with groups=sum(enable_lora)
            # Notes about shapes above
            # - self.lora_A has shape (4, 128): 4 because rank is 2 and LoRA is applied only to two matrices;
            # 128 is the input size of the x (embedding size). (4, 128) and not (128, 4) because later on in
            # F.linear function weights are automatically transposed. In addition conv1d requires channels to
            # be before seq length
            # - self.lora_B has shape (256, 2): 256 because LoRA applied only to two matrices, so the output is
            # 128*2; 2 tells to have two channels per group for group convolution

            # Scaling:
            # This balances the pretrained model`s knowledge and the new task-specific adaptation
            # https://lightning.ai/pages/community/tutorial/lora-llm/
            # So, set alpha to 1.0 to fully add LoRA. If the LoRA seems to have too much effect (i.e., overfitted), set
            # alpha to lower value. If the LoRA seems to have too little effect, set alpha to higher than 1.0. You can
            # tune these values to your needs. This value can be even slightly greater than 1.0!
            # https://github.com/cloneofsimo/lora
            self.scaling = self.lora_alpha / self.r

            # Freezing the pre-trained weight matrix
            self.weight.requires_grad = False  # (384, 128)

            # Compute the indices
            # Indices are needed to properly pad weight updates with zeros. If we want to finetune queires and values,
            # but not keys, then the weights update should be:
            #
            # [[ΔW,ΔW,ΔW, ..., 0,0,0, ..., ΔW,ΔW,ΔW,],
            #  [....................................],
            #  [ΔW,ΔW,ΔW, ..., 0,0,0, ..., ΔW,ΔW,ΔW,]]
            #      ↑              ↑            ↑
            # ________________________________________
            # | query         | key       | value    |
            # ----------------------------------------
            self.lora_ind = self.weight.new_zeros((out_features,), dtype=torch.bool).view(
                len(enable_lora),
                -1,
            )  # (3, 128)
            self.lora_ind[enable_lora, :] = True  # (3, 128)
            self.lora_ind = self.lora_ind.view(-1)  # (384,)

        self.reset_parameters()
        if fan_in_fan_out:
            self.weight.data = self.weight.data.T

    def reset_parameters(self) -> None:
        """Reset all the weights, even including pretrained ones."""
        nn.Linear.reset_parameters(self)
        if hasattr(self, "lora_A"):
            # initialize A the same way as the default for nn.Linear and B to zero
            # Wondering why 'a' and is equal to math.sqrt(5)?: https://github.com/pytorch/pytorch/issues/15314
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)

    def zero_pad(self, x: torch.Tensor) -> torch.Tensor:
        """Properly pad weight updates with zeros.

        If we want to finetune queires and values, but not keys, then the weights update should be:

        [[ΔW,ΔW,ΔW, ..., 0,0,0, ..., ΔW,ΔW,ΔW,],
         [....................................],
         [ΔW,ΔW,ΔW, ..., 0,0,0, ..., ΔW,ΔW,ΔW,]]
            ↑              ↑            ↑
        ________________________________________
        | query         | key       | value    |
        ----------------------------------------

        Returns
        -------
        torch.Tensor
            tensor with weight updates and zeros for diselected q, k or v
        """
        # Let's image that input x has shape of (64, 64, 256), and self.out_features=384,
        # where embedding_size=128, enable_lora=[True, False, True], then 256 because
        # weights are updated only for query and values (2 * 128) and pretrained weights
        # store for query, key and values (3 * 128)
        # Note: double transpose (in the beginning and in the end) is basically a guard for two-dimensional tensors
        # for example when we want to merge/unmerge LoRA weights and pretrained weights
        x = x.transpose(0, 1)
        result = x.new_zeros((*x.shape[:-1], self.out_features))  # (64, 64, 384)
        result = result.view(-1, self.out_features)  # (4096, 384)
        result[:, self.lora_ind] = x.reshape(
            -1,
            self.out_features // len(self.enable_lora) * sum(self.enable_lora),
        )  # (4096, 256)
        return result.view((*x.shape[:-1], self.out_features)).transpose(0, 1)  # (64, 64, 384)

    def train(self, mode: bool = True) -> None:
        """Set the module into train or eval mode if `mode` is True of False respectively.

        For train mode (train(True)) if weights are merged we need to subtract weights updates (LoRA_A @ LoRA_B) from
        pretrained weights so we can continue training LoRA's matrices A and B and keep pretrained weights frozen.

        For eval mode (train(False)) if weights are not merged we need to add weight updates to pretrained weights in
        order to reduce computational overhead.

        Parameters
        ----------
        mode : bool, optional
            if True the module will be set into train mode (affects Dropout and Batchnorm), if False - eval mode.

        """

        def T(w: torch.Tensor) -> torch.Tensor:  # noqa: N802
            return w.T if self.fan_in_fan_out else w

        # if train(True) -> in train mode if weights are merged then we need to unmerge them, otherwise do nothing
        # if train(False) -> in eval mode if weights are not merged we need to merge them, otherwise do nothing
        update_pretrained_weights = self.merged if mode else not self.merged

        # despite being called from nn.Linear this method will put
        # all layers into train mode, including nn.Dropout
        # of course except parameters (such as self.lora_A, self.lora_B)
        nn.Linear.train(self, mode)

        # Let's assume that:
        # ⚬ self.weight.data (384, 128) or (3 * embedding_size, embedding_size)
        # ⚬ self.lora_A.data (4, 128)
        # ⚬ self.lora_B.data (256, 2)
        if self.merge_weights and update_pretrained_weights:
            if self.r > 0 and any(self.enable_lora):
                delta_w = F.conv1d(
                    self.lora_A.data.unsqueeze(0),  # (4, 128) -> (1, 4, 128)
                    self.lora_B.data.unsqueeze(-1),  # (256, 2) -> (256, 2, 1)
                    groups=sum(self.enable_lora),
                ).squeeze(  # (1, 4, 128) @ (256, 2, 1) -> (1, 256, 128)
                    0,
                )  # (1, 256, 128) -> (256, 128)
                # -1: W = W - delta_W (unmerge), +1: W = W + delta_W (merge)
                sign = -1 if mode else 1
                self.weight.data += sign * self.zero_pad(T(delta_w * self.scaling))  # (256, 128) -> (384, 128)
            self.merged = not mode

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Do the forward pass.

        If LoRA's weights are merged with pretrained ones then it's a simple matrix multiplication.
        If not, then multiply pretrained weights with input, apply LoRA on input and do summation.

        Parameters
        ----------
        x : torch.Tensor
            input tensor of shape (batch_size, context_length, embedding_size)

        Returns
        -------
        torch.Tensor
            output tensor of shape (batch_size, context_length, 3 * embedding_size)
        """

        def T(w: torch.Tensor) -> torch.Tensor:  # noqa: N802
            return w.T if self.fan_in_fan_out else w

        # Let's assume that:
        # ⚬ x (64, 64, 128) or (batch_size, context_length, embedding_size)
        # ⚬ self.weight (384, 128) or (3 * embedding_size, embedding_size)
        # ⚬ self.lora_A.data (4, 128)
        # ⚬ self.lora_B.data (256, 2)

        # the logic here is that the weights are merged only during inferencing
        # so if they are merged we don't need to do anything with LoRA's A and B matrices
        # but if the weights are not merged that means that the forward method is called during
        # training and we need to forward pass input through pretrained weights, LoRA A and B matrices
        # and do the summation (as per scheme at the top of the file)
        if self.merged:
            return F.linear(x, T(self.weight), bias=self.bias)
        else:  # noqa: RET505
            # `F.linear` automatically transposes the second argument (T(self.weight) in our case)
            result = F.linear(x, T(self.weight), bias=self.bias)  # (64, 64, 128) @ (384, 128) -> (64, 64, 384)
            if self.r > 0:
                after_A = F.linear(  # noqa: N806
                    self.lora_dropout(x),
                    self.lora_A,
                )  # (64, 64, 128) @ (4, 128) -> (64, 64, 4)

                # For F.conv1d:
                # - input: input tensor of shape (minibatch,in_channels,iW)
                # - weight: filters of shape (out_channels,in_channels/groups,kW)
                # - groups: split input into groups, in_channels should be divisible by the number of groups. Default: 1
                # presumably iW - sequence width/length, kW - kernel width
                after_B = F.conv1d(  # noqa: N806
                    after_A.transpose(-2, -1),  # (64, 64, 4) -> (64, 4, 64)
                    self.lora_B.unsqueeze(-1),  # (256, 2) -> (256, 2, 1)
                    groups=sum(self.enable_lora),
                ).transpose(  # (64, 4, 64) @ (256, 2, 1) -> (64, 256, 64)
                    -2,
                    -1,
                )  # (64, 256, 64) -> (64, 64, 256)

                # (64, 64, 256) after zero_pad (64, 64, 384)
                result += self.zero_pad(after_B) * self.scaling
            return result


def mark_only_lora_as_trainable(model: nn.Module, bias: str = "none") -> None:
    """Freeze all modules except LoRA's and depending on 'bias' value unfreezes bias weights.

    Parameters
    ----------
    model : nn.Module
        model with LoRA layers
    bias : str, optional
        - `none`: all bias weights will be frozen
        - `lora_only`: only bias weight for LoRA layers will be unfrozen
        - `all`: all bias weights will be unfrozen

    Raises
    ------
    NotImplementedError
        raise if `bias` not in ["none", "lora_only", "all"]
    """
    # freeze all layers except LoRA's
    for n, p in model.named_parameters():
        if "lora_" not in n:
            p.requires_grad = False

    # depending on the `bias` value unfreeze bias weights
    if bias == "none":
        return
    elif bias == "all":  # noqa: RET505
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
    """Return state_dict with weight of LoRA's A and B matrices and with biases depending on the `bias` value.

    Parameters
    ----------
    model : nn.Module
        model with LoRA layers
    bias : str, optional
        - `none`: state dict will not store biases
        - `lora_only`: state dict will store biases only from LoRA layers
        - `all`: state dict will store all biases

    Returns
    -------
    Dict[str, torch.Tensor]
        weights and biases of LoRA layers

    Raises
    ------
    NotImplementedError
        raise if `bias` not in ["none", "lora_only", "all"]
    """
    my_state_dict = model.state_dict()
    if bias == "none":
        return {k: my_state_dict[k] for k in my_state_dict if "lora_" in k}
    elif bias == "all":  # noqa: RET505
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


class LoRACausalSelfAttention(attention.CausalSelfAttention):
    lora_config = None

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
        """Do the same as multi-head attention but with a single matrix multiplication.

        Instead of creating multiple heads and concatenating the result (in addition to creating separate matrices for
        query, key and value for each head) we can do this in a single pass with a single weight matrix.

        In addition this class uses Low Ranking Adaptation (LoRA) for efficient finetuning.

        Parameters
        ----------
        embeddings_size : int
            size of the embeddings - the size of input of self-attention
        context_size : int
            the number of tokens that will be used during calculation attention map and
            weighted averaging of value of each token
        head_size : Optional[int]
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
        super().__init__(
            embeddings_size=embeddings_size,
            context_size=context_size,
            head_size=head_size,
            num_heads=num_heads,
            bias=bias,
            dropout=dropout,
            is_decoder=is_decoder,
        )
        logger.debug("Using Causal Self Attention with LoRA")

        # replace Linear with MergedLinear
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


@contextmanager
def lora(r: int, alpha: int, dropout: float, enabled: bool = True) -> None:
    """Apply context manager under which you can instantiate the model with LoRA.

    Parameters
    ----------
    r : int
        rank of the weight update matrices. To make sense of using LoRA the rank should be smaller than the rank of
        the weights of the model.  The rank can be as low as 1: https://arxiv.org/pdf/2106.09685.pdf (section 7.2)
    alpha : int
        alpha is needed for scaling updates as alpha/r
        "This scaling helps to reduce the need to retune hyperparameters when we vary r"
        https://arxiv.org/pdf/2106.09685.pdf (section 4.1)
    dropout : float
        dropout that is applied on the input in the LoRA branch (before multiplying by matrix A)
    enabled : bool
        enables/disables LoRA
    """
    if not enabled:
        yield
        return

    # Note: in a nutshell the code below forces `transformer_block` to use LoRA variant of causal self-attention instead
    # of the original one (without LoRA)

    # set LoRA config
    LoRACausalSelfAttention.lora_config = LoRAConfig(r=r, alpha=alpha, dropout=dropout)
    # save original causal self-attention into a variable
    block_causal_self_attention = transformer_block.CausalSelfAttention
    # replace original causal self-attention with LoRA variant
    transformer_block.CausalSelfAttention = LoRACausalSelfAttention
    yield
    # reset original causal self-attention
    transformer_block.CausalSelfAttention = block_causal_self_attention
    # reset lora config
    LoRACausalSelfAttention.lora_config = None
