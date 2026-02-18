# Tensor (Model) Parallelism

**One-Line Summary**: Tensor parallelism splits individual layers of a neural network across multiple GPUs, so each GPU computes only a slice of every layer's output, enabling training of models whose single layers are too large for one device.

**Prerequisites**: Understanding of matrix multiplication, how transformer layers work (attention and feed-forward blocks), data parallelism and its memory limitations, basics of GPU interconnects (NVLink, PCIe).

## What Is Tensor Parallelism?

Imagine a restaurant kitchen where a single dish requires chopping an enormous pile of vegetables. Instead of one chef doing all the chopping (which would take too long), you divide the pile among four chefs, each working on their quarter simultaneously at the same counter. They are all working on the same dish at the same time, just handling different portions of the ingredients. Occasionally they need to pass partially prepared ingredients to each other (a quick handoff since they are shoulder-to-shoulder), and then continue.

Tensor parallelism applies this idea to neural network layers. Instead of one GPU computing the full output of a matrix multiplication, the weight matrix is partitioned across multiple GPUs. Each GPU multiplies the input by its slice of the weight matrix, producing a partial result. A quick communication step combines these partial results, and training proceeds to the next operation.

This is fundamentally different from data parallelism (where each GPU processes different data through the full model) and pipeline parallelism (where each GPU handles different layers). In tensor parallelism, every GPU works on the **same data** through the **same layer**, but each handles a different **slice of that layer's parameters**.

## How It Works

### Partitioning Strategies for Linear Layers

Consider a linear layer `Y = XW + b`, where `X` is the input of shape `[batch, d_in]` and `W` is the weight matrix of shape `[d_in, d_out]`.

**Column-wise (output) partitioning**: Split `W` along columns into `N` chunks:

```
W = [W_1 | W_2 | ... | W_N]     (each W_i is [d_in, d_out/N])
```

Each GPU `i` computes `Y_i = X * W_i`, producing a partial output of shape `[batch, d_out/N]`. The full output is the concatenation `Y = [Y_1 | Y_2 | ... | Y_N]`. Note that each GPU needs the **full input** `X` but produces only a **partial output**.

**Row-wise (input) partitioning**: Split `W` along rows into `N` chunks:

```
W = [W_1; W_2; ... ; W_N]     (each W_i is [d_in/N, d_out])
```

This requires splitting the input `X` along its last dimension as well: `X = [X_1 | X_2 | ... | X_N]`. Each GPU computes `Y_i = X_i * W_i`, producing a partial sum of shape `[batch, d_out]`. The full output requires an **all-reduce** (summation): `Y = Y_1 + Y_2 + ... + Y_N`.

### Application to Transformer Layers

Megatron-LM, the seminal framework for tensor parallelism, cleverly applies these two strategies in sequence to minimize communication.

**MLP Block**: A transformer MLP has two linear layers with a nonlinearity (GeLU) in between:

```
Y = GeLU(X * W_1) * W_2
```

- `W_1` is split **column-wise**: each GPU computes `GeLU(X * W_1_i)` independently. No communication is needed here because GeLU is element-wise and operates on each column partition independently.
- `W_2` is split **row-wise**: each GPU computes its partial result, and a single **all-reduce** produces the final output.

This design requires only **one all-reduce per MLP block** in the forward pass (and one in the backward pass).

**Self-Attention Block**: Attention heads distribute naturally across GPUs. With `h` attention heads and `N` GPUs, each GPU handles `h/N` heads:

- The `Q`, `K`, `V` projection matrices are split column-wise, with each GPU computing projections for its subset of heads.
- Each GPU independently computes attention for its assigned heads (no communication needed since heads are independent).
- The output projection matrix is split row-wise, and a single **all-reduce** combines the results.

Again, only **one all-reduce per attention block** per direction.

### Communication Volume

For a transformer layer with hidden dimension `d` on `N` GPUs, each all-reduce communicates approximately `2 * batch_size * seq_len * d` bytes per operation (using fp16/bf16). With two all-reduces per layer (one for attention, one for MLP) in each of the forward and backward passes, the total communication is:

```
Communication per layer = 4 * 2 * B * S * d bytes  (forward + backward, attention + MLP)
```

This must complete with very low latency, which is why tensor parallelism demands high-bandwidth interconnects.

## Why It Matters

Tensor parallelism is essential for training large language models because it addresses a fundamental constraint: individual layers can be too large for a single GPU's memory. A model with hidden dimension 12,288 (like GPT-3 175B) has weight matrices with over 150 million parameters per layer. When combined with optimizer states and activations, a single layer's memory footprint can approach or exceed a GPU's capacity.

Tensor parallelism also reduces the **per-GPU computation time** for each layer, which directly reduces the latency of each training step. Unlike data parallelism (which increases throughput but not per-step speed), tensor parallelism makes individual steps faster.

## Key Technical Details

- **NVLink is effectively required**: Tensor parallelism communicates at every layer boundary, multiple times per training step. On NVIDIA hardware, NVLink provides 600-900 GB/s bidirectional bandwidth (A100/H100), compared to ~64 GB/s for PCIe Gen4. Using tensor parallelism over PCIe is typically impractical due to the communication frequency.
- **Typical tensor parallel degree**: 2, 4, or 8 GPUs within a single node (matching NVLink topology). Going beyond 8 usually offers diminishing returns because communication overhead grows.
- **Hidden dimension must be divisible**: The model's hidden dimension `d` must be divisible by the tensor parallel degree `N`. Similarly, the number of attention heads `h` must be divisible by `N`. This is why LLM architectures often choose hidden dimensions that are powers of 2 or have many factors.
- **Activation memory reduction**: Since each GPU only computes a fraction of the activations, tensor parallelism reduces activation memory by roughly a factor of `N`.
- **Dropout and layer norm**: Operations like dropout must use the same random seed across tensor parallel ranks to maintain mathematical equivalence. LayerNorm typically operates on the full hidden dimension, requiring an all-reduce of statistics.

## Common Misconceptions

- **"Tensor parallelism and model parallelism are the same thing."** Model parallelism is a broader term that includes tensor parallelism (splitting within layers) and pipeline parallelism (splitting across layers). Tensor parallelism specifically refers to intra-layer splitting.
- **"Tensor parallelism scales across nodes."** In practice, it rarely does. The communication frequency is too high for inter-node bandwidths. Tensor parallelism is almost exclusively used within a single node connected by NVLink.
- **"More tensor parallel GPUs always means faster training."** Beyond a certain point (typically 8 GPUs), the communication overhead of additional all-reduces outweighs the computational savings. The computation per GPU becomes too small relative to the fixed communication latency.
- **"Tensor parallelism eliminates the need for data parallelism."** They solve different problems. Tensor parallelism addresses model size; data parallelism addresses training speed. Most large-scale setups use both.

## Connections to Other Concepts

- **Data Parallelism**: Tensor parallelism is typically used **within** a node, while data parallelism replicates across groups of nodes. They are complementary and almost always combined.
- **Pipeline Parallelism**: Pipeline parallelism splits across layers (inter-layer) while tensor parallelism splits within layers (intra-layer). They address the same problem (model too large for one GPU) from different angles.
- **3D Parallelism**: The combination of data, tensor, and pipeline parallelism, where tensor parallelism handles the intra-node dimension.
- **Sequence Parallelism**: Extends tensor parallelism to split the sequence dimension for operations like LayerNorm and dropout that otherwise require the full hidden state, further reducing activation memory.
- **Attention Mechanisms**: Multi-head attention is naturally amenable to tensor parallelism because attention heads operate independently, making the column-wise split communication-free during the core attention computation.

## Further Reading

- Shoeybi et al., *"Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism"* (2019) -- The foundational paper introducing efficient tensor parallelism for transformers with column/row partitioning strategies.
- Narayanan et al., *"Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM"* (2021) -- Extends Megatron-LM with detailed analysis of tensor parallelism's interaction with pipeline and data parallelism.
- Korthikanti et al., *"Reducing Activation Recomputation in Large Transformer Models"* (2022) -- Introduces sequence parallelism as a companion to tensor parallelism, addressing the remaining memory bottleneck from activations in non-tensor-parallel operations.
