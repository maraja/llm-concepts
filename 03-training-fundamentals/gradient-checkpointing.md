# Gradient Checkpointing

**One-Line Summary**: Gradient checkpointing trades additional computation for dramatically reduced memory during training by selectively storing activations at checkpoint layers and recomputing intermediate values during the backward pass.

**Prerequisites**: Backpropagation, forward and backward passes, activation memory, GPU memory, training large models

## What Is Gradient Checkpointing?

Imagine you are driving from New York to Los Angeles, and you need to remember every turn you made in order to describe the route to someone later. One approach is to record every single turn as you drive (expensive in storage but effortless to replay). Another approach is to save only a few major waypoints -- "I passed through Pittsburgh, then St. Louis, then Denver" -- and when asked for details, you re-drive each segment to reconstruct the turns between waypoints. You spend more time driving but need far less storage. Gradient checkpointing applies this same trade-off to neural network training: save only some intermediate results, re-derive the rest when needed.

During the forward pass of neural network training, the model computes activations at every layer. During the backward pass (backpropagation), these activations are needed to compute gradients. Standard training stores every activation in memory, which is fast but extremely memory-hungry. For a 70B parameter model, activation memory alone can exceed 120GB -- far more than the memory available on a single GPU. Gradient checkpointing solves this by storing activations at only a subset of layers (the "checkpoints") and recomputing the others from the nearest checkpoint during the backward pass.

This technique is not optional for large-scale training -- it is a necessity. Without gradient checkpointing, the activation memory requirements of modern LLMs would make training physically impossible on available hardware. Combined with mixed-precision training, FlashAttention, and ZeRO optimizer sharding, gradient checkpointing is one of the four pillars that make large-scale LLM training feasible.

## How It Works

### Standard Backpropagation Memory Usage

In standard training, the forward pass computes and stores activations at every layer:

```
Forward pass (standard):
Layer 1:  input -> a1  (store a1 in memory)
Layer 2:  a1 -> a2     (store a2 in memory)
Layer 3:  a2 -> a3     (store a3 in memory)
...
Layer L:  a_{L-1} -> a_L  (store a_L in memory)

Memory: O(L) -- all L layers' activations stored simultaneously
```

For a transformer with L layers, each layer's activations include the hidden states, attention matrices, and intermediate feed-forward values. For a large model:

```
Per-layer activation memory (approximate):
- Hidden states: batch_size * seq_len * hidden_dim * 2 bytes (fp16)
- Attention matrix: batch_size * num_heads * seq_len^2 * 2 bytes
- FFN intermediates: batch_size * seq_len * 4 * hidden_dim * 2 bytes

Example (70B model, batch=1, seq_len=4096):
- Per layer: ~1.5 GB
- Total (80 layers): ~120 GB  <-- exceeds single GPU memory!
```

### Checkpointed Backpropagation

With gradient checkpointing, only selected layers save their activations. During the backward pass, non-checkpointed activations are recomputed from the nearest upstream checkpoint:

```
Forward pass (checkpointed, every 4th layer):
Layer 1:  input -> a1  (CHECKPOINT -- store a1)
Layer 2:  a1 -> a2     (discard a2)
Layer 3:  a2 -> a3     (discard a3)
Layer 4:  a3 -> a4     (discard a4)
Layer 5:  a4 -> a5     (CHECKPOINT -- store a5)
...

Backward pass:
Need gradient at Layer 3?
  -> Recompute: a1 -> a2 -> a3 (from stored checkpoint a1)
  -> Compute gradient using recomputed a3
  -> Discard a2, a3 again
```

### Optimal Checkpoint Placement

The classic result from Chen et al. (2016) shows that checkpointing every sqrt(L) layers achieves an optimal balance:

```
Layers:       L = 80 (typical for a large model)
Checkpoints:  sqrt(80) ≈ 9 checkpoints

Memory:       O(sqrt(L)) instead of O(L)
              ~15 GB instead of ~120 GB

Compute overhead: ~33% (each non-checkpointed segment is recomputed once)
```

The 33% overhead comes from the fact that, on average, each activation is computed twice: once in the forward pass and once during recomputation in the backward pass. Since the forward pass is roughly half the total compute (backward is ~2x forward), recomputing the forward pass adds ~33% to total training time.

### Implementation in Practice

PyTorch provides built-in gradient checkpointing through `torch.utils.checkpoint`:

```python
import torch
from torch.utils.checkpoint import checkpoint

class TransformerBlock(torch.nn.Module):
    def __init__(self, ...):
        self.attention = MultiHeadAttention(...)
        self.ffn = FeedForwardNetwork(...)
        self.norm1 = LayerNorm(...)
        self.norm2 = LayerNorm(...)

    def forward(self, x):
        x = x + self.attention(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x

class CheckpointedTransformer(torch.nn.Module):
    def __init__(self, num_layers, ...):
        self.layers = torch.nn.ModuleList(
            [TransformerBlock(...) for _ in range(num_layers)]
        )

    def forward(self, x):
        for layer in self.layers:
            # Wrap each layer with checkpointing
            x = checkpoint(layer, x, use_reentrant=False)
        return x
```

In frameworks like DeepSpeed and Megatron-LM, checkpointing is configured at a higher level:

```python
# DeepSpeed configuration
{
    "activation_checkpointing": {
        "partition_activations": true,
        "contiguous_memory_optimization": true,
        "number_checkpoints": 9,  # sqrt(L) for 80 layers
        "synchronize_checkpoint_boundary": false
    }
}
```

### Advanced Strategies

**Selective Checkpointing**: Not all layers consume equal memory. Attention layers (with their O(n^2) attention matrices) are far more memory-intensive than feed-forward layers. Selective checkpointing saves attention outputs while recomputing only the cheaper components:

```
Memory savings breakdown:
- Checkpointing attention only: ~60% memory reduction, ~15% compute overhead
- Checkpointing everything:     ~85% memory reduction, ~33% compute overhead
- FlashAttention (no explicit attention matrix) + selective checkpointing:
  Optimal balance of memory and compute
```

**Offloading Checkpoints to CPU**: Instead of discarding non-checkpointed activations entirely, they can be offloaded to CPU RAM (which is much larger than GPU memory) and loaded back during the backward pass. This trades PCIe bandwidth for both memory and compute savings.

## Why It Matters

1. **Makes large model training physically possible**: Without gradient checkpointing, the activation memory of models like LLaMA 70B or GPT-4 would exceed the memory of any available hardware. Checkpointing is not an optimization -- it is a requirement.
2. **Enables larger batch sizes**: By reducing activation memory, checkpointing frees up GPU memory for larger batch sizes, which can improve training throughput and gradient quality.
3. **Modest compute cost for massive memory savings**: The ~33% compute overhead is a remarkably good trade-off for 8-10x memory reduction. In practice, the savings often enable configurations that would otherwise be impossible, not just marginally better.
4. **Universally adopted**: Every major LLM training framework (PyTorch, DeepSpeed, Megatron-LM, JAX/Pax, Fairscale) implements gradient checkpointing as a standard feature.
5. **Composable with other techniques**: Checkpointing combines naturally with mixed-precision training, FlashAttention, tensor parallelism, and ZeRO sharding to create the full memory-efficiency stack required for frontier model training.

## Key Technical Details

- The sqrt(L) checkpoint spacing minimizes the product of memory usage and compute overhead. Fewer checkpoints save more memory but increase recomputation; more checkpoints reduce recomputation but consume more memory.
- Compute overhead in practice is often less than the theoretical 33% because (a) not all operations are bottlenecked by compute (some are memory-bound), and (b) the recomputation can overlap with other operations.
- Gradient checkpointing does NOT change the mathematical result of training -- the gradients computed are bit-for-bit identical to standard backpropagation. It is a pure memory-compute trade-off with no impact on model quality.
- When combined with FlashAttention (which avoids materializing the full attention matrix), the memory savings are even more dramatic because the largest activation tensor (attention weights) is eliminated entirely.
- For a 70B model: full activations ~120GB, with checkpointing ~15GB, with checkpointing + FlashAttention ~8GB. This is the difference between impossible and feasible on 8xA100 (80GB each).
- Random number state must be carefully managed during recomputation to ensure dropout and other stochastic layers produce identical results on recomputation as on the original forward pass.
- Gradient checkpointing increases the backward pass time but does not affect the forward pass time, making it particularly suitable for training workloads where memory is the bottleneck rather than compute throughput.

## Common Misconceptions

- **"Gradient checkpointing approximates gradients."** The gradients are exact. Recomputed activations are mathematically identical to the original activations (assuming deterministic operations and proper random state management). There is zero impact on training quality.

- **"Gradient checkpointing doubles training time."** The overhead is approximately 33%, not 100%. This is because only the forward pass is recomputed, and the forward pass is roughly half the cost of the backward pass. The total overhead is one additional forward pass, added to the cost of one forward pass plus one backward pass.

- **"You only need gradient checkpointing for very large models."** Even medium-sized models benefit when training with long sequences, large batch sizes, or limited GPU memory. Researchers training 7B models on consumer GPUs (24GB) routinely use checkpointing to fit reasonable batch sizes.

- **"Gradient checkpointing is the same as model checkpointing."** These are completely different concepts. Model checkpointing saves the model weights to disk periodically for recovery from failures. Gradient checkpointing manages activation memory during a single training step. The shared word "checkpoint" is an unfortunate naming collision.

## Connections to Other Concepts

- **FlashAttention**: Eliminates the quadratic attention matrix, complementing gradient checkpointing by removing the single largest activation tensor.
- **Mixed-Precision Training (FP16/BF16)**: Halves the memory per activation value, stacking with gradient checkpointing for multiplicative memory savings.
- **ZeRO Optimizer Sharding**: Reduces optimizer state memory (a separate concern from activations), working alongside checkpointing to reduce total memory footprint.
- **Distributed Training (Tensor/Pipeline Parallelism)**: Splits activation memory across devices, combining with checkpointing when per-device memory is still insufficient.
- **Activation Recomputation**: Gradient checkpointing is the most common form of activation recomputation, but the general principle (recompute rather than store) applies to many settings beyond layer checkpoints.

## Further Reading

- Chen et al., "Training Deep Nets with Sublinear Memory Cost" (2016) -- the foundational paper establishing the sqrt(L) checkpointing strategy
- Korthikanti et al., "Reducing Activation Recomputation in Large Transformer Models" (2022) -- Megatron-LM's selective checkpointing strategies for transformers
- Rajbhandari et al., "ZeRO: Memory Optimizations Toward Training Trillion Parameter Models" (2020) -- DeepSpeed's memory optimization framework including checkpointing integration
- PyTorch Documentation, "torch.utils.checkpoint" -- practical implementation guide for gradient checkpointing in PyTorch
- Dao et al., "FlashAttention" (2022) -- complementary memory optimization that eliminates the attention activation bottleneck
