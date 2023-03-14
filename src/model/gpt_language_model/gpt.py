import math
from functools import reduce
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from loguru import logger
from torch import Tensor, nn
from tqdm import tqdm, trange

from src.model.gpt_language_model.transformer_block import LayerNorm, TransformerBlock
from src.utils.seed import set_seed


class GPTLanguageModel(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embeddings_size: int,
        context_size: int,
        head_size: Optional[int],
        num_heads: int,
        feed_forward_scaling: int,
        num_layers: int,
        bias: bool,
        dropout: float,
        weight_tying: bool = True,
        weight_decay: Optional[float] = None,
    ) -> None:
        """Create Generative Pre-trained Transformer model (decoder part of transformer architecture).

        Parameters
        ----------
        vocab_size : int
            number of unique tokens in vocabulary
        embeddings_size : int
            size of the embeddings - the size of input of self-attention
        context_size : int
            the number of tokens that will be used during calculation attention map and
            weighted averaging of value of each token
        head_size : Optional[int]
            the size of output of self-attention
        num_heads : int
            how many self-attention heads to use
        feed_forward_scaling : int
            feed-forward has two fully-connected layers; the number of neurons between them is larger
            than input and output sizes, `feed_forward_scaling` specifies by how much
        num_layers : int
            how many transformer blocks to use
        bias: bool
            whether to use bias or not: without bias might be a bit better and faster
        dropout : float
            how many connection between tokens are dropped during each forward pass
        weight_tying: bool
           Weight Tying improves the performance of language models by tying (sharing) the weights of the embedding and
           softmax layers. This method also massively reduces the total number of parameters in the language models that
           it is applied to.
           https://paperswithcode.com/method/weight-tying, by default True
        weight_decay: Optional[float]
            if provided will prepare parameters for optimizer
        """
        super().__init__()

        self.vocab_size = vocab_size
        self.embeddings_size = embeddings_size
        self.context_size = context_size
        self.head_size = head_size
        self.num_heads = num_heads
        self.feed_forward_scaling = feed_forward_scaling
        self.num_layers = num_layers
        self.bias = bias
        self.dropout = dropout
        self.weigh_tying = weight_tying
        self.weight_decay = weight_decay

        self.token_embedding_table = nn.Embedding(self.vocab_size, self.embeddings_size)
        # since attention doesn't have any notion of space and we want to use spatial information we need to implement
        # positional embeddings (they will encode relative position of each token)
        # positional embeddings knows how to encode position of last N (context_size) tokens
        self.positional_embedding_table = nn.Embedding(self.context_size, self.embeddings_size)
        self.embeddings_dropout = nn.Dropout(self.dropout)
        self.transformer_blocks = nn.Sequential(
            *[
                TransformerBlock(
                    embeddings_size=self.embeddings_size,
                    context_size=self.context_size,
                    head_size=self.head_size,
                    num_heads=self.num_heads,
                    bias=self.bias,
                    dropout=self.dropout,
                    feed_forward_scaling=self.feed_forward_scaling,
                    is_decoder=True,
                    use_causal_self_attention=True,
                )
                for _ in range(self.num_layers)
            ],
        )
        self.layer_norm_final = LayerNorm(self.embeddings_size, bias=self.bias)  # final layer norm
        self.language_model_head = nn.Linear(self.embeddings_size, self.vocab_size, bias=False)
        if self.weigh_tying:
            self.token_embedding_table.weight = self.language_model_head.weight

        self.apply(self.__init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for name, param in self.named_parameters():
            if name.endswith("projection.weight"):
                torch.nn.init.normal_(param, mean=0.0, std=0.2 / math.sqrt(2 * self.num_layers))

        # configure parameters for optimizer that will be decay and that will not
        if self.weight_decay:
            self.optimizer_parameters = self.__optimizer_parameters(weight_decay=self.weight_decay)

        # report number of parameters
        logger.debug(
            "GPT language model is created with number of parameters: {:.2f} million".format(
                self.__get_parameters_number() / 1e6,
            ),
        )

    def __get_parameters_number(self, exclude_positional_embeddings: bool = True) -> int:
        """Return total number of parameters of the model without counting parameters of positional embeddings."""
        params_count = sum(param.numel() for param in self.parameters())
        if exclude_positional_embeddings:
            params_count -= self.positional_embedding_table.weight.numel()
        return params_count

    def __init_weights(self, module: "torch.nn.modules") -> None:
        """Initialize Embedding and Linear layers with a smaller std.

        By default weights of Embedding layer are initialized from normal distribution
        with zero mean and unit std ( N(0, 1) ).
        Weights for linear layer are initialized from uniform distribution from
        [-k**0.5, k**0.5], where k = 1 / in_features. Even with 128 features it will be
        [-0.09, 0.09].

        Parameters
        ----------
        module : torch.nn.modules
            module of the network
        """
        if isinstance(module, (nn.Embedding, nn.Linear)):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if hasattr(module, "bias") and module.bias is not None:
                torch.nn.init.zeros_(module.bias)

    def __optimizer_parameters(self, weight_decay: float) -> Tuple[dict, dict]:
        """Configure optimizer with weight decay for nn.Linear.

        Parameters
        ----------
        weight_decay : float
            weight decay (L2 penalty)

        Returns
        -------
        Tuple[dict, dict]
            list of two dictionaries, containing parameter names that will have weight decay
            and that will not accordingly

        Raises
        ------
        ValueError
            if model's layer are not in (nn.Linear, nn.LayerNorm, LayerNorm, nn.Embedding)
        """
        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay, no_decay = set(), set()
        expected_weight_modules = (nn.Linear, nn.LayerNorm, LayerNorm, nn.Embedding)
        for pn, _ in self.named_parameters():
            # get the parent module by the parameter's name
            module = reduce(lambda module, key: getattr(module, key), pn.split(".")[:-1], self)
            if type(module) not in expected_weight_modules:
                msg = f"Expected the module to be one of '{expected_weight_modules}', but got {type(module)}"
                raise ValueError(msg)
            if isinstance(module, nn.Linear) and pn.endswith("weight"):
                decay.add(pn)
            else:
                no_decay.add(pn)

        # create the pytorch optimizer object
        param_dict = dict(self.named_parameters())
        return [
            {"params": [param_dict[pn] for pn in sorted(decay)], "weight_decay": weight_decay},
            {"params": [param_dict[pn] for pn in sorted(no_decay)], "weight_decay": 0.0},
        ]

    def forward(self, idx: Tensor, *, inference: bool = False) -> Tensor:
        """Do the whole forward pass for decoder part of transformer.

        This forward method includes all steps for decoder:
        1. token embeddings + positional
        2. transformer block consisting of self-attention, feed-forward, addNorm
        3. logits for each token in vocabulary (or the last one in case of inference)

        Parameters
        ----------
        idx : Tensor
            tensor of size (batch, time-step) consisting of indices of tokens inside vocabulary
            for each time-step for each batch
        inference: bool
            during inference we don't care about all tokens but the very last one, so we can
            apply final language head only on the last token and save some computations

        Raises
        ------
        ValueError
            if there is a mismatch between number of time-steps and self.context_size

        Returns
        -------
        Tensor
            tensor of size (batch, time-step, vocabulary_size): logits for each token in vocabulary
            for each time-step for each batch, or the last one in case of inference
        """
        # batch, time-step
        B, T = idx.shape  # noqa: N806
        if self.context_size < T:
            msg = f"Cannot do forward pass on sequence of length {T}, "
            f"context size should less or equal to {self.context_size}"
            raise ValueError(msg)

        # obtain token embeddings and add positional information
        token_embeddings = self.token_embedding_table(idx)  # (B, T, C)
        positional_embeddings = self.positional_embedding_table.weight[:T]  # (T, C)
        x = token_embeddings + positional_embeddings  # (B, T, C)
        x = self.embeddings_dropout(x)  # (B, T, C)

        # apply multiple transformer blocks
        x = self.transformer_blocks(x)  # (B, T, C)
        # apply final normalization and generate logits for each token in vocabulary
        x = self.layer_norm_final(x)  # (B, T, C)

        # during inference we don't need to encode all token predictions,
        # only the last one (newly generated)
        if inference:
            return self.language_model_head(x[:, -1:, :])  # (B, 1, vocab_size)
        return self.language_model_head(x)  # (B, T, vocab_size)

    def loss(self, logits: Tensor, targets: Tensor) -> Tensor:
        """Prepare tensors to be compatible with Pytorch's Cross-Entropy loss and applies it.

        Cross-Entropy expects to have features as the second-dimension, so we first need to
        transform logits in supported shape and in order to align targets tensor with logits tensor,
        we need to transform targets too.

        Parameters
        ----------
        logits : Tensor
            tensor with model's outputs
        targets : Tensor
            tensor with true labels

        Returns
        -------
        Tensor
            tensor with loss value (of how good model's predictions are)
        """
        B, T, C = logits.shape  # noqa: N806
        return F.cross_entropy(
            logits.view(B * T, C),
            targets.view(B * T),
        )

    @classmethod
    def from_pretrained(cls, gpt2_size):
        # huggingface transformers library is needed only in this method
        from transformers import GPT2Config, GPT2LMHeadModel

        # check that the gpt2 size is supported
        supported_sizes = ("gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl")
        if gpt2_size not in supported_sizes:
            msg = f"Only '{supported_sizes}' are supported, but '{gpt2_size}' was provided."
            logger.error(msg)
            raise ValueError(msg)

        gpt2_hf_config = GPT2Config.get_config_dict(gpt2_size)[0]
        # syncing argument names between our GPT implementation and from Huggingface
        gpt_config = {
            "vocab_size": gpt2_hf_config["vocab_size"],
            "embeddings_size": gpt2_hf_config["n_embd"],
            "context_size": gpt2_hf_config["n_ctx"],
            "num_layers": gpt2_hf_config["n_layer"],
            "num_heads": gpt2_hf_config["n_head"],
            "bias": True,
        }

        dropout_set = {value for name, value in gpt2_hf_config.items() if "dropout" in name}
        if len(dropout_set) != 1:
            msg = "All dropouts for GPT2 model should have had the same value, "
            f"but in fact recieved '{dropout_set}'"
            logger.error(msg)
            raise ValueError(msg)
        gpt_config["dropout"] = list(dropout_set)[0]
        gpt_config["head_size"] = None
        gpt_config["feed_forward_scaling"] = 4

        logger.debug("Creating GPT model with parameters: {}".format(gpt_config))
        model = GPTLanguageModel(**gpt_config)
        logger.debug("Model is created.")

        # extract gpt model parameters into a variable
        gpt_state_dict = model.state_dict()
        # drop tril as it's a buffer (doesn't learn anything)
        gpt_state_dict_keys = [k for k in gpt_state_dict.keys() if not k.endswith(".self_attention.tril")]

        # create Huggingface pretrained GPT2 model
        logger.debug("Loading pretrained Huggingface model of size '{}' ...".format(gpt2_size))
        gpt2_hf_model = GPT2LMHeadModel.from_pretrained(gpt2_size)
        logger.debug("Huggingface model is loaded.")
        gpt2_hf_state_dict = gpt2_hf_model.state_dict()
        # skip bias as it's not a parameter
        gpt2_hf_state_dict_keys = [
            key
            for key in gpt2_hf_state_dict.keys()
            if not key.endswith(
                (
                    ".attn.bias",
                    ".attn.masked_bias",
                )
            )
        ]

        if len(gpt_state_dict_keys) != len(gpt2_hf_state_dict_keys):
            msg = f"Mismatch number of keys between: {len(gpt_state_dict_keys)} != {len(gpt2_hf_state_dict_keys)}"
            logger.error(msg)
            raise ValueError(msg)

        # since names of layers are different for our implementation and the one from Huggingface,
        # we need to map them properly
        param_mapping = {
            "transformer": "",
            "wte": "token_embedding_table",
            "wpe": "positional_embedding_table",
            "h": "transformer_blocks",
            "attn": "self_attention",
            "c_attn": "causal_self_attention",
            "ln_1": "layer_norm_1",
            "ln_2": "layer_norm_2",
            "c_proj": "projection",
            "mlp": "feed_forward",
            "ln_f": "layer_norm_final",
            "lm_head": "language_model_head",
        }

        def sync_name(name):
            names_renamed = [param_mapping.get(n, n) for n in name.split(".")]
            return ".".join(filter(None, names_renamed))

        # in Huggingface implementation attention and feed forward uses 'Conv1D', while in this implementation,
        # that means that we can use weights for those layers, only we need to transpose them
        to_transposed = ("attn.c_attn.weight", "attn.c_proj.weight", "mlp.c_fc.weight", "mlp.c_proj.weight")

        # loading weights
        logger.debug("Starting copying weights from pretrained Huggingface model into our implementation ...")
        # TODO: rename source and destination
        for key in gpt2_hf_state_dict_keys:
            key_dst = sync_name(key)
            src_weights = gpt2_hf_state_dict[key]
            if key.endswith(to_transposed):
                src_weights = src_weights.t()
            if src_weights.shape != gpt_state_dict[key_dst].shape:
                msg = f"Shape mismatch: shape of source '{src_weights.shape}' and destinatiion - '{gpt_state_dict[key_dst].shape}'"
                logger.error(msg)
                raise ValueError(msg)
            with torch.no_grad():
                gpt_state_dict[key_dst].copy_(src_weights)
        logger.debug("Weights are copied.")

        return model

    @torch.no_grad()
    def generate(
        self,
        idx: Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k_logits: Optional[int] = None,
    ) -> Tensor:
        """Generate new character after the current one.

        Parameters
        ----------
        idx : Tensor
            index of the current character
        max_new_tokens : int
            number of characters to be generated
        temperature : float, optional
            The temperature determines how greedy the generative model is:
            If the temperature is low, the probabilities to sample other but the class with the highest log probability
            will be small, and the model will probably output the most correct text, but rather boring, with small
            variation.
            If the temperature is high, the model can output, with rather high probability, other words than those with
            the highest probability. The generated text will be more diverse, but there is a higher possibility of
            grammar mistakes and generation of nonsense.
            https://ai.stackexchange.com/questions/32477/what-is-the-temperature-in-the-gpt-models, by default 1.0
        top_k_logits : Optional[int], optional
            only top K logits (with the highest value) will be kept, by default None

        Returns
        -------
        Tensor
            tensor containing indices of the provided characters and newly generated
        """
        # idx is (B, T) array of indices in the current context
        for _ in trange(max_new_tokens, ascii=True):
            # crop idx to the last block_size items
            context = idx[:, -self.context_size :]
            # get the predictions
            logits = self(context, inference=True)  # (B, T, C), with inference=True -> (1, 1, C)
            # focus only on the last time step and scale by desired temperature
            logits = logits[:, -1, :] / temperature  # becomes (B, C)
            if top_k_logits:
                # topk returns rearranged tensor where the first column contains the highest values,
                # the last column - the smallest values from top K logits ...
                values, _ = torch.topk(logits, min(top_k_logits, logits.shape[-1]))
                # ... that's why we need to compare with the last column
                logits[logits < values[:, -1]] = float("-inf")  # `-1:` is to preserve dimensionality
            # apply softmax on the predictions to get probabilities
            probs = F.softmax(logits, dim=-1)  # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T + 1)
        return idx
