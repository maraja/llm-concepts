# 3D Parallelism & Training at Scale

**One-Line Summary**: 3D parallelism combines data, tensor, and pipeline parallelism into a unified strategy that maps each dimension to the hardware topology, enabling the training of the largest language models (hundreds of billions to trillions of parameters) across thousands of GPUs.

**Prerequisites**: Understanding of data parallelism and gradient synchronization, tensor parallelism and intra-layer splitting, pipeline parallelism and micro-batching, GPU cluster topology (nodes, NVLink, InfiniBand), ZeRO/FSDP memory optimization concepts.

## What Is 3D Parallelism?

Imagine building a skyscraper. You need three kinds of organization simultaneously: (1) within each floor, specialized teams handle different sections of the same floor plan in parallel (this is tensor parallelism -- splitting work within a single layer); (2) different floors are assigned to different construction crews who pass materials up the building sequentially (this is pipeline parallelism -- splitting consecutive layers); (3) you build multiple identical buildings at once to house more people (this is data parallelism -- processing different data through identical model replicas).

![3D parallelism topology mapping data, tensor, and pipeline parallelism to hardware hierarchy](https://jalammar.github.io/images/model-parallelism/3d-parallelism-megatron.png)
*Source: [Jay Alammar - The Illustrated Model Parallelism](https://jalammar.github.io/model-parallelism/)*


No single strategy alone can handle the scale of training frontier language models. A model with 175 billion to over a trillion parameters, trained on trillions of tokens across thousands of GPUs for months, requires the orchestrated combination of all three parallelism dimensions. 3D parallelism is the engineering framework that makes this possible.

## How It Works


*Recommended visual: Diagram showing how D x T x P GPUs are organized with tensor parallelism within nodes, pipeline parallelism across nodes, and data parallelism across replicas -- see [Megatron-LM paper (Narayanan et al., 2021)](https://arxiv.org/abs/2104.04473), Figure 3*

### The Three Dimensions

Given a cluster of `D * T * P` total GPUs, 3D parallelism creates:

- **T-way tensor parallelism**: `T` GPUs within a node split individual layers, communicating via NVLink at each layer boundary.
- **P-way pipeline parallelism**: `P` groups of `T` GPUs (possibly across nodes) each hold a contiguous shard of the model's layers, communicating activation tensors between stages.
- **D-way data parallelism**: `D` identical pipeline replicas process different data, synchronizing gradients via all-reduce across replicas.

Total GPUs = `D * T * P`

### Mapping to Hardware Topology

The critical insight is that each parallelism dimension has different communication characteristics, and modern GPU clusters have a hierarchical bandwidth structure:

| Level | Bandwidth | Parallelism Mapped |
|-------|-----------|-------------------|
| Within GPU (HBM) | ~3 TB/s | Computation |
| Within node (NVLink) | 600-900 GB/s | Tensor parallelism |
| Across nodes (InfiniBand) | 50-100 GB/s | Pipeline parallelism |
| Across racks (network) | 10-50 GB/s | Data parallelism |

**Tensor parallelism** communicates at every layer (highest frequency, smallest messages) and therefore demands the highest bandwidth. It maps to the intra-node NVLink mesh.

**Pipeline parallelism** communicates only at stage boundaries (lower frequency, medium-sized activation tensors) and uses point-to-point communication. It maps to the inter-node InfiniBand fabric.

**Data parallelism** communicates once per training step (lowest frequency, but largest total volume for gradient all-reduce). It can tolerate the lowest bandwidth tier because communication can overlap with computation.

### Concrete Example: Training a 175B Parameter Model

Consider training on 1,024 A100 GPUs (128 nodes, 8 GPUs per node):

- **Tensor parallel degree (T) = 8**: All 8 GPUs within each node form a tensor-parallel group. Individual layers are split 8-way.
- **Pipeline parallel degree (P) = 16**: 16 consecutive nodes (each running 8-way tensor parallelism) form a pipeline, with each node holding ~6 of the model's 96 layers.
- **Data parallel degree (D) = 8**: The 1,024 GPUs form 8 complete pipeline replicas, each processing different data shards.

Verification: `D * T * P = 8 * 8 * 16 = 1,024 GPUs`.

Each training step:
1. Data is split 8 ways across pipeline replicas.
2. Each replica splits its data into micro-batches for pipeline scheduling.
3. Within each pipeline stage, the 8 GPUs split the layer computation via tensor parallelism.
4. After all micro-batches complete, gradient all-reduce synchronizes across the 8 data-parallel replicas.

### Beyond 3D: Additional Parallelism Dimensions

Modern frontier model training often extends to 4D or 5D parallelism:

**Expert Parallelism (for Mixture-of-Experts models)**:
MoE models have sparse expert layers where each token is routed to only a few experts. Expert parallelism places different experts on different GPUs. During computation, tokens are routed to their assigned experts via **all-to-all** communication. This adds a fourth parallelism dimension:

```
Total GPUs = D * T * P * E
```

where `E` is the expert-parallel degree. MoE models like Mixtral, GShard, and Switch Transformer use this to scale parameter count without proportionally scaling compute.

**Sequence Parallelism**:
For very long context windows (32K, 128K, or 1M+ tokens), the activation memory scales linearly with sequence length and can dominate GPU memory. Sequence parallelism splits the sequence dimension across GPUs:

- **Megatron-style sequence parallelism**: Splits non-tensor-parallel operations (LayerNorm, dropout) along the sequence dimension, complementing tensor parallelism. This is a memory optimization within the tensor-parallel group.
- **Ring Attention / Context Parallelism**: Distributes the attention computation across GPUs, where each GPU holds a chunk of the key-value cache and attention is computed in a ring-style fashion. This enables arbitrarily long sequences.

**Context Parallelism**:
A specialized form of sequence parallelism (used in Megatron-LM and LLaMA 3.1 training) that splits the sequence across GPUs specifically for the attention operation. Each GPU computes attention for its chunk of queries against all keys/values (communicated in a ring pattern).

### Scheduling and Coordination

With 3D+ parallelism, the training loop becomes a carefully orchestrated dance:

1. **Data distribution**: Global batch distributed across data-parallel ranks, split into micro-batches for pipeline scheduling.
2. **Pipeline scheduling**: 1F1B or interleaved schedule coordinates forward and backward passes across pipeline stages.
3. **Tensor-parallel communication**: Within each pipeline stage's forward and backward compute, all-reduce operations synchronize partial results across the tensor-parallel group.
4. **Gradient synchronization**: After all micro-batches complete, all-reduce across data-parallel ranks.
5. **Optimizer step**: Each rank updates its parameters (possibly sharded via ZeRO/FSDP within the data-parallel dimension).

### Real-World Infrastructure Requirements

Training frontier models demands extraordinary infrastructure:

*Recommended visual: PTD-P (Pipeline, Tensor, Data Parallelism) schedule showing micro-batch interleaving across pipeline stages with tensor-parallel groups -- see [Lilian Weng's blog post on Large Transformer Model Training](https://lilianweng.github.io/posts/2021-09-25-train-large/)*


- **GPU count**: 2,000-16,000+ GPUs (H100 or newer)
- **Interconnect**: Multi-rail InfiniBand (400-3200 Gbps per node) or proprietary interconnects (Google TPU pods, NVLink Switch)
- **Training duration**: 3-6 months continuous operation
- **Power consumption**: 10-30+ MW for a training cluster
- **Failure handling**: With thousands of GPUs running for months, hardware failures are frequent (every few hours). Robust checkpointing (every 10-30 minutes) and automatic restart mechanisms are essential.
- **Storage**: Petabytes of training data, high-throughput distributed file systems (Lustre, GPFS) to feed data to thousands of GPUs simultaneously.

## Why It Matters

3D parallelism is not an academic curiosity -- it is the only practical way to train frontier language models. GPT-4, Claude, Gemini, LLaMA, and every model at the 70B+ parameter scale uses some form of multi-dimensional parallelism. Understanding 3D parallelism is essential for anyone who wants to grasp how these models are actually built, not just how they work architecturally.

The choice of parallelism configuration directly impacts training throughput (and therefore cost). A well-tuned 3D parallelism setup on 1,024 GPUs can achieve 40-55% Model FLOPs Utilization (MFU), meaning the GPUs spend 40-55% of their time on actual useful computation. Poor configuration can drop this below 20%, effectively doubling or tripling the cost and time of training.

## Key Technical Details

- **Model FLOPs Utilization (MFU)**: The gold-standard efficiency metric. It measures what fraction of theoretical peak GPU FLOPs is spent on useful model computation (excluding communication, pipeline bubbles, and recomputation). Frontier labs report 40-55% MFU.
- **Communication overlap**: All three parallelism dimensions offer opportunities to overlap communication with computation. Effective overlap is the difference between 30% and 50% MFU.
- **Parallelism configuration search**: Finding the optimal (D, T, P) configuration is non-trivial. It depends on model size, sequence length, batch size, hardware topology, and interconnect bandwidth. Tools like Megatron-LM's configuration planner and Alpa's automated parallelism optimizer help.
- **ZeRO within data parallelism**: The data-parallel dimension can use ZeRO Stage 1 or 2 to further reduce optimizer state memory without the communication overhead of full sharding.
- **Activation checkpointing**: Nearly universally used at scale. Recomputing activations during the backward pass trades ~33% extra compute for 60-70% activation memory savings.

## Common Misconceptions

- **"You always need all three dimensions."** For models under ~30B parameters, data parallelism + tensor parallelism (2D) is often sufficient. Pipeline parallelism adds complexity and bubble overhead, so it is only introduced when necessary.
- **"3D parallelism configuration is one-size-fits-all."** The optimal configuration is highly specific to the model architecture, hardware, and training hyperparameters. Changing the sequence length, batch size, or even the number of available nodes can shift the optimal (D, T, P) split.
- **"More GPUs always means faster training."** Beyond a certain point, communication overhead, pipeline bubbles, and reduced per-GPU efficiency mean that adding GPUs yields diminishing returns. There is a practical sweet spot for every model size.
- **"Frontier model training is just about having enough GPUs."** The software engineering challenge is at least as significant as the hardware. Communication libraries, fault tolerance, checkpointing, data pipelines, and training stability all require deep expertise.

## Connections to Other Concepts

- **Data Parallelism (DDP)**: The simplest parallelism dimension, always present in 3D parallelism, responsible for scaling training throughput.
- **Tensor Parallelism**: The intra-node dimension, critical for splitting large layers that exceed single-GPU memory.
- **Pipeline Parallelism**: The inter-node model-splitting dimension, enabling models to span across nodes with manageable communication.
- **ZeRO / FSDP**: Often used as the data-parallel backend, adding memory efficiency to the data-parallel dimension.
- **Mixed Precision Training**: Universal at scale. FP16/BF16 computation with FP32 master weights reduces both memory and communication volume.
- **MoE (Mixture of Experts)**: Adds the expert parallelism dimension, enabling models with trillions of parameters while keeping per-token compute manageable.
- **Flash Attention**: Reduces activation memory for the attention mechanism, complementing parallelism strategies by lowering the per-GPU memory floor.

## Further Reading

- Narayanan et al., *"Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM"* (2021) -- The definitive guide to 3D parallelism, presenting the mapping of tensor/pipeline/data parallelism to hardware topology with detailed scaling analysis.
- Smith et al., *"Using DeepSpeed and Megatron to Train Megatron-Turing NLG 530B"* (2022) -- Describes the practical 3D parallelism configuration used to train one of the largest dense language models, combining DeepSpeed ZeRO with Megatron-LM's tensor and pipeline parallelism.
- Dubey et al., *"The Llama 3 Herd of Models"* (2024) -- Details Meta's training infrastructure for the LLaMA 3 family (up to 405B parameters), including their 4D parallelism strategy (tensor + pipeline + data + context parallelism) on 16,384 H100 GPUs.
