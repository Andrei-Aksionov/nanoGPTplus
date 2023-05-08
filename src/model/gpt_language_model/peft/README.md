# Parameter-Efficient Finetuning for LLMs

There are two approaches of calculating query, key and value matrices:

In the basic implementation of attention mechanism you have three separate weight matrices for query, key and
value and thus in order to obtain q, k, v you apply these 3 matrices separately on input x (in self-attention).
This approach is covered with LoRA.Linear class. It's not implemented here, but one can find implementation here:
<https://github.com/microsoft/LoRA/blob/main/loralib/layers.py#L91>

The other approach is to have a single matrix that stores weights for all three matrices (query, key and value).
So you can apply this big combined matrix ones (helps with parallelization on GPU) and then you can split the
output into three chunks to have queries, keys and values. To cover this approach MergedLinear class is created.

Note: examples of calculating qkv matrices with separate multiplications and a single one you can find in
`attention.py` files of this repository.
