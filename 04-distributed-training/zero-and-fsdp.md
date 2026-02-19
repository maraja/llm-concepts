# ZeRO & FSDP (Fully Sharded Data Parallel)

**One-Line Summary**: ZeRO and FSDP eliminate the memory redundancy of data parallelism by sharding optimizer states, gradients, and parameters across GPUs, enabling training of models that no single GPU can hold while preserving the simplicity of data-parallel training.

**Prerequisites**: Solid understanding of data parallelism and all-reduce, knowledge of how optimizers work (especially Adam's memory footprint), awareness of GPU memory constraints and what consumes VRAM during training.

## What Is ZeRO / FSDP?

In standard data parallelism, every GPU holds a complete copy of everything: the model parameters, the gradients, and the optimizer states. This is enormously wasteful. If you have 64 GPUs, you are storing 64 identical copies of the optimizer states, 64 identical copies of the gradients, and 64 identical copies of the model parameters. The only thing that differs across GPUs is the data.

Imagine a library with 64 branches, each keeping a full copy of every book, every catalog card, and every librarian's reading notes. ZeRO's insight is: why not distribute the collection? Each branch keeps only 1/64th of the books, notes, and catalogs. When a patron needs a book held by another branch, it is quickly delivered, used, and returned. The total storage across the system is the same as a single branch, not 64 times that.

**ZeRO** (Zero Redundancy Optimizer), developed by Microsoft Research for DeepSpeed, implements this in three progressive stages. **FSDP** (Fully Sharded Data Parallel), developed by Meta for PyTorch, brings the ZeRO Stage 3 concept natively into the PyTorch ecosystem.

## How It Works

### The Memory Problem

For a model with `P` parameters trained in mixed precision with the Adam optimizer, the per-GPU memory consumption in standard data parallelism is:

| Component | Precision | Memory |
|-----------|-----------|--------|
| Parameters | fp16/bf16 | 2P bytes |
| Gradients | fp16/bf16 | 2P bytes |
| Optimizer: master weights | fp32 | 4P bytes |
| Optimizer: momentum (Adam) | fp32 | 4P bytes |
| Optimizer: variance (Adam) | fp32 | 4P bytes |
| **Total** | | **16P bytes** |

For a 7B parameter model: `16 * 7 * 10^9 = 112 GB` -- already exceeding the 80GB of an A100. And this does not even include activations.

The key observation: in standard data parallelism across `N` GPUs, the total memory used system-wide is `16P * N` bytes, while only `16P` bytes of unique data exists. The redundancy factor is `N`.

### ZeRO Stages

**ZeRO Stage 1: Shard Optimizer States**

Each GPU stores only `1/N`th of the optimizer states (master weights, momentum, variance). After the backward pass and gradient all-reduce, each GPU updates only its shard of the parameters using its shard of the optimizer states. An all-gather then broadcasts the updated parameters to all GPUs.

Memory per GPU: `4P + 12P/N` bytes (full parameters + full gradients + sharded optimizer states)

For 64 GPUs, this reduces optimizer state memory by ~64x, saving roughly 75% of total memory.

**ZeRO Stage 2: Shard Optimizer States + Gradients**

In addition to Stage 1, gradients are also partitioned. Instead of an all-reduce (which leaves all gradients on all GPUs), a **reduce-scatter** operation is used: each GPU ends up with only the gradient shard it needs for its optimizer state partition.

Memory per GPU: `2P + 2P/N + 12P/N` bytes (full parameters + sharded gradients + sharded optimizer states)

This eliminates gradient redundancy as well, saving roughly 87.5% of total memory for large `N`.

**ZeRO Stage 3: Shard Everything (Optimizer States + Gradients + Parameters)**

Parameters themselves are also sharded. Each GPU stores only `1/N`th of the model parameters. Before each forward or backward computation for a layer, the full parameters for that layer are gathered via **all-gather** from all GPUs. After the computation, the non-local parameters are discarded.

Memory per GPU: `16P/N` bytes (everything sharded equally)

This achieves the theoretical minimum: total memory is divided equally across all GPUs, with zero redundancy. A 175B parameter model that would require 2.8TB in standard data parallelism can fit across 64 GPUs with only ~44GB per GPU (before activations).

### FSDP: ZeRO Stage 3 in PyTorch

PyTorch's FSDP implements the ZeRO Stage 3 concept as a native PyTorch module wrapper. Key mechanics:

1. **Sharding**: When a model is wrapped with FSDP, each parameter is sharded across the data-parallel group. Each GPU stores only its `1/N`th shard as a flat, contiguous tensor.

2. **Forward pass**: Before a layer's forward method executes, FSDP triggers an **all-gather** to reconstruct the full parameters from all shards. After the forward pass, the non-local parameters are freed (unless needed for backward, controlled by the `forward_prefetch` and `backward_prefetch` options).

3. **Backward pass**: Similar all-gather operations reconstruct parameters needed for gradient computation. After computing gradients, a **reduce-scatter** distributes gradient shards back to their owning GPUs.

4. **Optimizer step**: Each GPU updates only its local shard of parameters using its local shard of gradients and optimizer states.

5. **Communication-computation overlap**: FSDP prefetches the next layer's parameters during the current layer's computation, hiding much of the all-gather latency.

### Communication Cost Analysis

ZeRO Stage 3 / FSDP requires more communication than standard DDP:

- **DDP**: One all-reduce per step = `2P` bytes transmitted per GPU
- **ZeRO-3/FSDP**: All-gather (forward) + all-gather (backward) + reduce-scatter (backward) = `3P` bytes transmitted per GPU (approximately 1.5x DDP)

In practice, with effective prefetching and overlap, the throughput difference between FSDP and DDP is often only 5-15%.

## Why It Matters

ZeRO and FSDP fundamentally changed the economics of large model training. Before ZeRO, the only options for training models larger than one GPU's memory were tensor parallelism and pipeline parallelism, both of which require significant code changes and careful hardware-aware configuration. ZeRO/FSDP allows researchers to scale model size with the same familiar data-parallel programming model: wrap your model, choose your sharding strategy, and train.

This democratization means that a team with 8 GPUs can now train models that previously required complex custom distributed frameworks. FSDP's integration into PyTorch's core library has made it the default approach for many organizations training models in the 7B-70B parameter range.

## Key Technical Details

- **Sharding strategies in FSDP**: `FULL_SHARD` (ZeRO-3), `SHARD_GRAD_OP` (ZeRO-2), and `NO_SHARD` (standard DDP). Choosing the right strategy depends on model size and GPU count.
- **FSDP wrapping granularity**: FSDP can wrap at different levels (entire model, per-transformer-block, per-layer). Finer wrapping reduces peak memory but increases communication calls.
- **Mixed precision with FSDP**: Parameters can be sharded in fp32 but gathered in fp16/bf16 for computation, reducing communication volume while maintaining optimizer precision.
- **Activation checkpointing**: Complementary to ZeRO/FSDP. Since activation memory is not sharded by ZeRO (each GPU's activations are unique to its data), checkpointing is often needed for very large models.
- **CPU offloading (ZeRO-Infinity)**: ZeRO Stage 3 can offload shards to CPU memory or even NVMe storage, enabling training of trillion-parameter models on limited GPU hardware at the cost of throughput.

## Common Misconceptions

- **"FSDP is a completely different approach from ZeRO."** FSDP is essentially ZeRO Stage 3 implemented natively in PyTorch. The core algorithmic idea (sharding all three components across data-parallel ranks) is identical.
- **"ZeRO Stage 3 / FSDP makes training much slower due to extra communication."** The overhead is approximately 1.5x in communication volume compared to DDP, but with prefetching and overlap, the actual throughput reduction is typically 5-15%. The memory savings far outweigh this cost.
- **"ZeRO eliminates the need for tensor and pipeline parallelism."** For the very largest models (100B+ parameters), ZeRO/FSDP alone may not be sufficient because (a) the all-gather overhead at extreme scale becomes significant, and (b) activation memory (which ZeRO does not shard) may still exceed GPU capacity. Tensor and pipeline parallelism remain necessary for frontier-scale training.
- **"All three ZeRO stages should always be used."** Stage 1 or 2 may be sufficient and more efficient if the model fits in memory with just optimizer state/gradient sharding. Each additional stage increases communication, so use the minimum stage that meets your memory requirements.

## Connections to Other Concepts

- **Data Parallelism (DDP)**: ZeRO/FSDP is an evolution of data parallelism. It preserves the data-parallel training semantics (each GPU sees different data, gradients are averaged) while eliminating memory redundancy.
- **Tensor Parallelism**: Addresses a different dimension. ZeRO shards temporally (parameters are gathered, used, and discarded), while tensor parallelism shards spatially (each GPU permanently holds part of each layer). They can be combined.
- **Pipeline Parallelism**: Also complementary. In a 3D parallelism setup, FSDP can serve as the data-parallel component while pipeline and tensor parallelism handle the model-parallel dimensions.
- **Mixed Precision Training**: Tightly integrated with ZeRO/FSDP. The distinction between fp16 compute copies and fp32 optimizer master copies is central to the memory accounting.
- **Gradient Accumulation**: Often combined with FSDP to increase effective batch size without increasing per-micro-batch memory, allowing the all-gather communication to be amortized over more compute.

## Diagrams and Visualizations

![ZeRO Stages 1, 2, and 3 showing progressive sharding of optimizer states, gradients, and parameters](https://jalammar.github.io/images/model-parallelism/zero-deepspeed.png)
*Source: [Jay Alammar - The Illustrated Model Parallelism](https://jalammar.github.io/model-parallelism/)*

*Recommended visual: Memory consumption comparison across ZeRO stages showing the reduction from 16x redundancy to zero redundancy -- see [Rajbhandari et al., "ZeRO: Memory Optimizations Toward Training Trillion Parameter Models" (2020)](https://arxiv.org/abs/1910.02054), Figure 1*

*Recommended visual: FSDP all-gather and reduce-scatter communication pattern during forward and backward passes -- see [Lilian Weng - How to Train Really Large Models on Many GPUs](https://lilianweng.github.io/posts/2021-09-25-train-large/)*

## Further Reading

- Rajbhandari et al., *"ZeRO: Memory Optimizations Toward Training Trillion Parameter Models"* (2020) -- The foundational paper introducing ZeRO Stages 1-3, with detailed memory and communication analysis.
- Zhao et al., *"PyTorch FSDP: Experiences on Scaling Fully Sharded Data Parallel"* (2023) -- Describes PyTorch's FSDP implementation, design decisions, and scaling results up to thousands of GPUs.
- Rajbhandari et al., *"ZeRO-Infinity: Breaking the GPU Memory Wall for Extreme Scale Deep Learning"* (2021) -- Extends ZeRO Stage 3 with CPU and NVMe offloading, enabling training of models with trillions of parameters on limited GPU clusters.
