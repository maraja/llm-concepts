# Pipeline Parallelism

**One-Line Summary**: Pipeline parallelism distributes consecutive layers of a model across different GPUs like an assembly line, using micro-batching to keep all stages busy simultaneously and minimize idle time (pipeline bubbles).

**Prerequisites**: Understanding of neural network forward and backward passes, familiarity with data parallelism and why models may not fit on a single GPU, basic intuition for latency vs. throughput trade-offs.

## What Is Pipeline Parallelism?

Consider an automobile assembly line with four stations: frame welding, engine installation, painting, and interior fitting. Each station handles one phase of production, and cars move sequentially from one station to the next. If only one car is on the line, three stations sit idle at any given moment. But if you feed multiple cars into the line in quick succession, all four stations can work simultaneously on different cars at different stages of completion.

Pipeline parallelism applies this assembly-line metaphor to neural network training. The model's layers are divided into consecutive **stages**, each assigned to a different GPU. A training batch is split into smaller **micro-batches** that flow through the pipeline in sequence. While GPU 4 runs the forward pass on micro-batch 1, GPU 3 can process micro-batch 2, GPU 2 handles micro-batch 3, and GPU 1 works on micro-batch 4 -- all simultaneously.

The challenge, as with any pipeline, is the **bubble**: the startup and drain phases where not all stages are active. Minimizing this bubble is the central design problem of pipeline parallelism.

## How It Works

### Basic Setup

Given a model with `L` layers and `P` pipeline stages (GPUs), each stage holds approximately `L/P` consecutive layers. During training:

1. The input micro-batch enters Stage 1, which computes the forward pass through its layers and sends the resulting activations to Stage 2.
2. Stage 2 receives the activations, continues the forward pass through its layers, and sends results to Stage 3.
3. This continues until Stage P produces the loss.
4. The backward pass then flows in reverse: Stage P computes its gradients and sends activation gradients back to Stage P-1, and so on.

### The Pipeline Bubble Problem

With a single micro-batch, only one GPU is active at any time. With `P` stages and `M` micro-batches, the pipeline bubble fraction is:

```
Bubble fraction = (P - 1) / (M + P - 1)
```

For this to be negligible, you need `M >> P`. For example, with 4 pipeline stages and 32 micro-batches, the bubble fraction is approximately 9%. With only 4 micro-batches, it balloons to 43%.

### Scheduling Strategies

**GPipe (Google, 2019)**:
- Execute all `M` micro-batch forward passes first, then all `M` backward passes.
- Simple to implement but requires storing activations for all `M` micro-batches simultaneously, leading to high memory usage.
- Bubble fraction: `(P - 1) / M` of total time.

```
GPU 1: F1 F2 F3 F4 -------- B4 B3 B2 B1
GPU 2:    F1 F2 F3 F4 -------- B4 B3 B2 B1
GPU 3:       F1 F2 F3 F4 -------- B4 B3 B2 B1
GPU 4:          F1 F2 F3 F4 -- B4 B3 B2 B1
```

(Dashes represent idle bubble time.)

**1F1B (One Forward, One Backward) Schedule**:
- After the pipeline fills (warmup phase), each GPU alternates between one forward and one backward micro-batch.
- Memory advantage: each GPU only needs to store activations for at most `P` micro-batches (not `M`), because backward passes consume activations as fast as forward passes produce them.
- Same bubble fraction as GPipe: `(P - 1) / M`, but dramatically less memory.

```
GPU 1: F1 F2 F3 F4 B1 F5 B2 F6 B3 ... B_M
GPU 2:    F1 F2 F3 B1 F4 B2 F5 B3 ...    B_M
GPU 3:       F1 F2 B1 F3 B2 F4 B3 ...       B_M
GPU 4:          F1 B1 F2 B2 F3 B3 ...          B_M
```

**PipeDream and Interleaved Schedules**:
- **PipeDream-Flush**: Similar to 1F1B but ensures weight consistency (all micro-batches in a batch use the same weights).
- **Interleaved Pipeline**: Each GPU is assigned multiple non-consecutive stages (e.g., GPU 1 handles stages 1 and 5, GPU 2 handles stages 2 and 6). This reduces the bubble by a factor equal to the number of stages per GPU (the "virtual pipeline stages" or `v`):

```
Bubble fraction = (P - 1) / (v * M + P - 1)
```

The trade-off is more frequent, smaller communications.

**Zero-Bubble Pipeline Parallelism**:
- Recent research achieves near-zero bubble by carefully reordering forward passes, backward-for-input passes (`B`), and backward-for-weight passes (`W`), exploiting the fact that weight gradient computation can be deferred.
- Approaches theoretical efficiency limits at the cost of more complex scheduling.

### Gradient Synchronization

Within a pipeline-parallel group, no gradient all-reduce is needed (each stage has unique layers). However, pipeline parallelism is typically combined with data parallelism, where gradient all-reduce occurs across pipeline replicas after all micro-batches in a batch complete.

## Why It Matters

Pipeline parallelism enables training models across multiple nodes where inter-node bandwidth is limited. Unlike tensor parallelism (which requires NVLink-class bandwidth due to per-layer communication), pipeline parallelism only communicates **activation tensors between adjacent stages** at layer boundaries. These activations are typically much smaller than the full gradient tensors, and the communication is point-to-point rather than collective. This makes pipeline parallelism well-suited for the inter-node dimension of a training cluster, where bandwidth may be 100-400 Gb/s InfiniBand rather than 600+ GB/s NVLink.

Pipeline parallelism also divides model memory across stages. A model with 96 layers split across 8 stages places only 12 layers on each GPU, reducing per-GPU parameter and activation memory proportionally.

## Key Technical Details

- **Activation communication**: Between stages, only the activation tensor at the layer boundary is transmitted. For a transformer with hidden dim `d`, batch size `b`, and sequence length `s`, this is `b * s * d` elements (typically in bf16), communicated point-to-point.
- **Micro-batch size trade-off**: Smaller micro-batches reduce the bubble but increase the ratio of communication-to-computation and may reduce GPU compute efficiency (smaller matrix multiplications).
- **Memory for activations**: GPipe stores activations for all `M` micro-batches; 1F1B stores at most `P`. Activation checkpointing (recomputation) can further reduce this at the cost of ~33% additional compute.
- **Load balancing**: Stages should have roughly equal computation time. The first and last stages often have additional work (embedding, loss computation), requiring careful layer assignment.
- **Batch size constraint**: The total batch size must be divisible into enough micro-batches (`M >> P`) to amortize the bubble. This can constrain hyperparameter choices.

## Common Misconceptions

- **"Pipeline parallelism has no idle time."** It always has a bubble at the start and end of each batch. The 1F1B and interleaved schedules minimize but never fully eliminate it (though zero-bubble approaches come very close).
- **"Pipeline parallelism is like data parallelism but for layers."** They are fundamentally different. Data parallelism processes different data through the same layers; pipeline parallelism processes the same data through different layers at different times.
- **"The bubble is always a major problem."** With sufficient micro-batches (M >= 4P to 8P), the bubble fraction drops below 5-10%, which is often acceptable. The real cost is the constraint on batch size and micro-batch count.
- **"Pipeline parallelism and tensor parallelism are interchangeable."** They have different communication patterns and hardware requirements. Tensor parallelism needs high-bandwidth intra-node links; pipeline parallelism works over lower-bandwidth inter-node connections. They are complementary, not substitutes.

## Connections to Other Concepts

- **Tensor Parallelism**: Often used together. Tensor parallelism handles intra-node splitting (within layers), while pipeline parallelism handles inter-node splitting (across layers). This combination leverages the hardware topology: fast NVLink within nodes, slower InfiniBand between nodes.
- **Data Parallelism**: Pipeline parallelism is almost always combined with data parallelism. Multiple pipeline replicas process different data subsets, with gradient synchronization across replicas.
- **3D Parallelism**: The full combination of data + tensor + pipeline parallelism used for the largest models.
- **Activation Checkpointing**: Particularly important in pipeline parallelism to reduce the memory overhead of stored activations across micro-batches.
- **ZeRO / FSDP**: Can be combined with pipeline parallelism to shard optimizer states across data-parallel ranks within each pipeline stage.

## Diagrams and Visualizations

![GPipe pipeline parallelism schedule showing forward and backward passes with pipeline bubbles](https://jalammar.github.io/images/model-parallelism/gpipe-bubble.png)
*Source: [Jay Alammar - The Illustrated Model Parallelism](https://jalammar.github.io/model-parallelism/)*

*Recommended visual: Comparison of GPipe vs 1F1B pipeline schedules showing how 1F1B reduces memory requirements while maintaining the same bubble fraction -- see [Lilian Weng - How to Train Really Large Models on Many GPUs](https://lilianweng.github.io/posts/2021-09-25-train-large/)*

*Recommended visual: Interleaved pipeline schedule with virtual stages, showing reduced bubble fraction -- see [Narayanan et al., "Efficient Large-Scale Language Model Training on GPU Clusters" (2021)](https://arxiv.org/abs/2104.04473), Figure 8*

## Further Reading

- Huang et al., *"GPipe: Efficient Training of Giant Neural Networks using Pipeline Parallelism"* (2019) -- Introduces pipeline parallelism with micro-batching for neural network training, establishing the foundational concepts.
- Narayanan et al., *"Memory-Efficient Pipeline-Parallel DNN Training"* (PipeDream-2BW / PipeDream-Flush, 2021) -- Addresses weight consistency issues in asynchronous pipeline schedules and introduces the 1F1B schedule used in practice.
- Qi et al., *"Zero Bubble Pipeline Parallelism"* (2023) -- Achieves near-zero pipeline bubbles by separating backward computation into input-gradient and weight-gradient phases with novel scheduling algorithms.
