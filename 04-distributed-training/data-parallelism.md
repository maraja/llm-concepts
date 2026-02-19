# Data Parallelism & Distributed Data Parallel (DDP)

**One-Line Summary**: Data parallelism replicates the entire model on every GPU and splits the training data across them, synchronizing gradients after each step to keep all copies in lockstep.

**Prerequisites**: Understanding of SGD and mini-batch gradient descent, basic knowledge of neural network forward and backward passes, familiarity with GPU memory concepts and what a gradient is.

## What Is Data Parallelism?

Imagine you have a massive textbook to summarize and four friends willing to help. Rather than giving each friend a different chapter (which would require coordination about narrative flow), you photocopy the entire textbook for each friend and assign each person a different quarter of the pages to summarize. At the end, everyone shares their notes and you merge them into a single, complete summary.

![Data parallelism overview showing model replicas processing different data shards with gradient synchronization](https://jalammar.github.io/images/model-parallelism/data-parallelism.png)
*Source: [Jay Alammar - The Illustrated Model Parallelism](https://jalammar.github.io/model-parallelism/)*


Data parallelism works exactly this way. Every GPU gets a complete copy of the model. The training dataset is split into chunks, and each GPU processes its own chunk independently. After computing gradients on their local data, all GPUs communicate to average those gradients. Then every GPU applies the same averaged gradient update, keeping all model copies perfectly synchronized.

This is the simplest and most widely-used form of distributed training. If your model fits on a single GPU and you just want to train faster, data parallelism is almost always the right first choice.

## How It Works


*Recommended visual: Ring all-reduce algorithm showing how gradient chunks are passed around a ring of GPUs in 2(N-1) steps -- see [Lilian Weng - How to Train Really Large Models on Many GPUs](https://lilianweng.github.io/posts/2021-09-25-train-large/)*

### Step-by-Step Breakdown

1. **Initialization**: The model is replicated identically across `N` GPUs. All copies start with the same weights.

2. **Data Distribution**: A global mini-batch of size `B` is divided into `N` micro-batches of size `B/N`. Each GPU receives a unique micro-batch via a `DistributedSampler` that ensures no data overlap.

3. **Forward Pass**: Each GPU independently computes the forward pass on its local micro-batch, producing local losses.

4. **Backward Pass**: Each GPU independently computes gradients with respect to its local loss. At this point, GPU `i` holds gradients `g_i` computed only from its local data.

5. **Gradient Synchronization (All-Reduce)**: This is the critical communication step. All GPUs participate in an **all-reduce** operation that computes the average gradient across all replicas:

   ```
   g_avg = (1/N) * (g_1 + g_2 + ... + g_N)
   ```

   After the all-reduce, every GPU holds the identical averaged gradient `g_avg`. This is mathematically equivalent to computing the gradient on the full mini-batch `B`.

6. **Parameter Update**: Each GPU applies the optimizer step using `g_avg`, producing identical updated weights across all replicas.

### The All-Reduce Operation

All-reduce is the workhorse of data parallelism. The most common implementation is the **ring all-reduce** algorithm:

- GPUs are arranged in a logical ring.
- Each GPU sends a chunk of its gradient tensor to its neighbor while simultaneously receiving a chunk from its other neighbor.
- After `2(N-1)` communication steps, every GPU has the complete sum.
- Total data transferred per GPU: `2 * (N-1)/N * M` bytes, where `M` is the model size in bytes. This scales nearly independently of `N`.

### PyTorch DDP: Overlapping Communication with Computation

Naive data parallelism waits for the entire backward pass to finish before starting the all-reduce. PyTorch's **DistributedDataParallel (DDP)** is smarter. It:

- Groups parameters into **buckets** (default ~25MB each).
- As soon as gradients for a bucket are computed during the backward pass, the all-reduce for that bucket begins **immediately**, overlapping with the ongoing gradient computation for earlier layers.
- Since backpropagation proceeds from the last layer to the first, the last layer's gradients are ready first and can be communicated while earlier layers are still computing.

This overlap can hide a significant fraction of communication latency, making DDP substantially faster than naive implementations.

### Scaling Efficiency

In the ideal case, training throughput scales linearly with the number of GPUs:

*Recommended visual: DDP bucketed gradient all-reduce overlapping with backward pass computation -- see [PyTorch DDP documentation](https://pytorch.org/docs/stable/notes/ddp.html)*


```
Throughput_N = N * Throughput_1
```

In practice, scaling efficiency is:

```
Efficiency = Throughput_N / (N * Throughput_1)
```

Typical DDP efficiency on modern hardware:
- 2-8 GPUs within a node (NVLink): 95-99% efficiency
- 16-64 GPUs across nodes (InfiniBand): 85-95% efficiency
- 100+ GPUs across nodes: 70-90% efficiency, depending on model size and network bandwidth

Larger models tend to scale better because the computation-to-communication ratio improves: gradient computation grows with model parameters, but the communication volume also grows proportionally, and larger models have more computation to overlap with that communication.

## Why It Matters

Data parallelism is the foundation of virtually all distributed training. Even when training the largest models, data parallelism is almost always one component of the overall strategy. It is straightforward to implement, introduces minimal code changes (often just a few wrapper lines in PyTorch), and provides near-linear speedups. For any model that fits on a single GPU, data parallelism alone can reduce training time from weeks to hours.

## Key Technical Details

- **Learning rate scaling**: When using `N` GPUs with data parallelism, the effective batch size is `N` times larger. The common practice is to scale the learning rate linearly: `lr_new = lr_base * N`, combined with a warm-up period to stabilize early training.
- **Batch normalization**: Standard BatchNorm computes statistics per-GPU. For true equivalence to single-GPU training, **SyncBatchNorm** communicates statistics across GPUs, though this adds overhead.
- **Random seed management**: Each GPU must use a different random seed for data augmentation and dropout, but the same seed for weight initialization.
- **Gradient accumulation**: If the per-GPU micro-batch is too small, you can accumulate gradients over multiple forward-backward passes before the all-reduce, simulating a larger effective batch.
- **Communication backends**: NCCL (NVIDIA Collective Communications Library) is the standard for GPU-to-GPU communication. Gloo is a fallback for CPU or non-NVIDIA setups.

## Common Misconceptions

- **"Data parallelism reduces memory per GPU."** It does not. Each GPU holds a full copy of the model, optimizer states, and activations. The only thing split is the data. Memory per GPU is essentially the same as single-GPU training (minus the slightly smaller micro-batch activations).
- **"Doubling GPUs halves training time."** In theory, yes. In practice, communication overhead, synchronization barriers, and reduced per-GPU batch sizes mean you get less than a 2x speedup. The efficiency gap widens with more GPUs.
- **"All-reduce is a bottleneck."** With modern interconnects (NVLink within nodes, InfiniBand across nodes) and DDP's overlap of communication with computation, all-reduce is often not the bottleneck. Data loading and preprocessing can be the actual limiting factor.
- **"Larger batch sizes are always better."** Very large batch sizes can degrade model quality. There is a critical batch size beyond which increasing batch size yields diminishing returns or hurts convergence. This limits how far data parallelism alone can scale.

## Connections to Other Concepts

- **Tensor Parallelism**: When the model is too large for one GPU, tensor parallelism splits individual layers across GPUs. Data parallelism and tensor parallelism are often combined.
- **Pipeline Parallelism**: Another approach for large models that splits consecutive layers across GPUs. Combined with data parallelism in 3D parallelism strategies.
- **ZeRO / FSDP**: These techniques address the memory limitation of data parallelism by sharding optimizer states, gradients, and parameters across GPUs while preserving the data-parallel training paradigm.
- **Mixed Precision Training**: Often used alongside data parallelism to reduce communication volume (sending fp16 gradients instead of fp32) and speed up computation.
- **Gradient Compression**: Techniques like gradient quantization or sparsification reduce communication volume in data parallelism, improving scaling efficiency at the cost of some approximation.

## Further Reading

- Goyal et al., *"Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour"* (2017) -- Establishes linear learning rate scaling rules and warm-up strategies for large-batch data-parallel training.
- Li et al., *"PyTorch Distributed: Experiences on Accelerating Data Parallel Training"* (2020) -- The official paper describing PyTorch DDP's design, including bucketed all-reduce and computation-communication overlap.
- Ben-Nun and Hoefler, *"Demystifying Parallel and Distributed Deep Learning: An In-Depth Concurrency Analysis"* (2019) -- Comprehensive survey of parallelism strategies in deep learning, with rigorous analysis of data parallelism's communication costs.
