# Mixed Precision Training

**One-Line Summary**: Mixed precision training uses lower-precision number formats (FP16 or BF16) for most computations while maintaining a master copy of weights in FP32, cutting memory usage in half and dramatically increasing throughput by leveraging specialized hardware tensor cores.

**Prerequisites**: Binary representation of numbers, basic understanding of floating-point formats, GPU architecture basics (what tensor cores are), the training loop (forward pass, backward pass, optimizer step), why memory is a bottleneck in LLM training.

## What Is Mixed Precision Training?

Imagine you are an architect designing a building. For your final blueprints, you need precise measurements down to the millimeter (FP32 -- full precision). But when doing rough sketches and quick calculations to explore designs, you only need measurements to the nearest centimeter (FP16/BF16 -- half precision). You save enormous time and paper doing the bulk of your work at lower precision, while keeping a precise master copy for the critical details.

Mixed precision training applies this same principle to neural network training. The "mixed" refers to using multiple numerical precisions simultaneously: lower precision for the bulk of computation (forward pass, backward pass, gradient computation) and higher precision where it matters most (optimizer state, parameter master copy).

## How It Works

### Floating-Point Formats Explained

A floating-point number is stored as three components: a **sign bit**, **exponent bits** (determining range), and **mantissa/fraction bits** (determining precision).

| Format | Total Bits | Exponent | Mantissa | Range | Precision |
|--------|-----------|----------|----------|-------|-----------|
| FP32 | 32 | 8 bits | 23 bits | ~$\pm 3.4 \times 10^{38}$ | ~7 decimal digits |
| FP16 | 16 | 5 bits | 10 bits | ~$\pm 65,504$ | ~3.3 decimal digits |
| BF16 | 16 | 8 bits | 7 bits | ~$\pm 3.4 \times 10^{38}$ | ~2.4 decimal digits |

**FP32 (Single Precision)**: The traditional default. Ample range and precision, but uses 4 bytes per value and does not benefit from tensor core acceleration.

**FP16 (Half Precision)**: Half the memory, but with a critical limitation: the small exponent (5 bits) means a maximum value of ~65,504. Gradients and activations that exceed this range cause **overflow** (becoming infinity), while very small gradients **underflow** to zero. This limited dynamic range makes raw FP16 training fragile.

**BF16 (Brain Floating Point)**: Developed by Google Brain specifically for deep learning. The key insight: BF16 keeps the same 8-bit exponent as FP32 (preserving the full range) but reduces the mantissa to only 7 bits. You lose some precision in the least significant digits, but you never encounter the overflow/underflow problems of FP16.

### Why Lower Precision Is Faster

Modern GPUs (NVIDIA A100, H100) contain **tensor cores** -- specialized hardware units designed for matrix multiplication in lower precision formats. The speedups are substantial:

- **FP16/BF16 tensor core operations**: 2-8x faster than FP32 operations on the same hardware.
- **Memory bandwidth**: Half the bytes means twice as many values can be moved between GPU memory and compute units per second. Since LLM training is often **memory-bandwidth bound**, this alone can nearly double throughput.
- **Memory capacity**: Storing activations, gradients, and parameters in half precision doubles the effective GPU memory, allowing larger batch sizes or longer sequences.

### The Mixed Precision Training Recipe

The standard mixed precision training procedure (introduced by Micikevicius et al., 2018):

1. **Maintain a master copy of weights in FP32.** This is stored in the optimizer and used for the actual parameter updates.
2. **Cast weights to FP16/BF16 for the forward pass.** All matrix multiplications, attention computations, and most activations use half precision.
3. **Compute loss in FP32.** The final loss computation and any reduction operations use full precision to avoid numerical issues.
4. **Backward pass in FP16/BF16.** Gradients are computed in half precision.
5. **Cast gradients to FP32 for the optimizer step.** Adam's moment estimates and the parameter update are performed in full precision.
6. **Update the FP32 master weights.** The optimizer modifies the FP32 copy, which is then cast back to FP16/BF16 for the next forward pass.

This workflow ensures that the cumulative effect of many small gradient updates is captured in full FP32 precision (preventing "gradient drift"), while the expensive matrix multiplications benefit from half-precision speed.

### The Loss Scaling Trick (FP16-Specific)

FP16's limited range creates a specific problem: many gradient values are small enough to underflow to zero, effectively losing gradient information. **Loss scaling** addresses this:

1. Before the backward pass, multiply the loss by a large scaling factor $S$ (e.g., $S = 1024$ or dynamically adjusted).
2. All gradients are now scaled by $S$, pushing small values into the representable FP16 range.
3. After the backward pass, divide gradients by $S$ before the optimizer step.

$$\text{scaled\_loss} = S \cdot \mathcal{L}$$
$$\text{scaled\_gradients} = \nabla_\theta(S \cdot \mathcal{L}) = S \cdot \nabla_\theta \mathcal{L}$$
$$\text{gradients} = \frac{\text{scaled\_gradients}}{S}$$

**Dynamic loss scaling** starts with a large $S$ and halves it whenever an overflow (inf/NaN) is detected, then gradually increases it when training is stable. PyTorch's `GradScaler` implements this automatically.

### Why BF16 Won

BF16 has become the dominant format for LLM training for one decisive reason: **it does not need loss scaling**. Because BF16 has the same exponent range as FP32, gradients never overflow or underflow due to range limitations. This eliminates an entire class of training instabilities and simplifies the training pipeline.

The reduced mantissa precision of BF16 (7 bits vs. FP16's 10 bits) means slightly less precision per value. However, in practice, neural network training is remarkably tolerant of low-precision arithmetic because:
- Individual rounding errors are random and tend to cancel out across large tensor operations.
- The stochastic nature of SGD inherently tolerates noise.
- The FP32 master copy captures long-term precision needs.

Starting with Google's TPUs (which natively supported BF16) and extending to NVIDIA's A100 and H100 GPUs, BF16 is now the standard choice for LLM training.

### Memory Savings Breakdown

For a model with $N$ parameters, the memory requirements are:

| Component | FP32 Training | Mixed Precision (BF16 + FP32 master) |
|-----------|--------------|--------------------------------------|
| Model parameters | $4N$ bytes | $2N$ bytes (BF16 for forward/backward) |
| Gradients | $4N$ bytes | $2N$ bytes (BF16) |
| Optimizer (Adam: m + v) | $8N$ bytes | $8N$ bytes (kept in FP32) |
| Master weights | N/A | $4N$ bytes (FP32 copy) |
| **Total** | **$16N$ bytes** | **$16N$ bytes** |

Wait -- the total looks the same! The savings come from **activations**, which dominate memory for large models. Activations stored in BF16 use half the memory of FP32 activations, and for long sequences with large batch sizes, this can save hundreds of gigabytes. Additionally, some implementations store optimizer states in lower precision or use techniques like 8-bit Adam to further reduce memory.

## Why It Matters

Without mixed precision training, training frontier LLMs would require roughly twice as many GPUs (or twice as long) and would cost billions more. It is one of the key engineering techniques that makes modern-scale LLM training economically feasible.

BF16 in particular removed a major source of training instability (FP16 overflow/underflow), making training runs more reliable and reducing the costly restarts that plagued earlier efforts. When a single training run costs $50-100 million, eliminating failure modes is worth enormous engineering effort.

## Key Technical Details

- **Certain operations must remain in FP32**: Softmax, layer normalization, and loss computation are typically performed in FP32 even during the mixed-precision forward pass, because they involve reductions that are sensitive to precision.
- **BF16 is not universally supported**: Older GPUs (pre-Ampere, i.e., before A100) do not have BF16 tensor cores. FP16 with loss scaling is used on V100s and older hardware.
- **FP8 is emerging**: NVIDIA's H100 and newer chips support FP8 (8-bit floating point) formats, promising another 2x speedup. FP8 training is an active research area as of 2024-2025.
- **Tensor cores require specific alignment**: Matrix dimensions typically need to be multiples of 8 (for FP16/BF16) or 16 (for FP8) to fully utilize tensor cores. This influences model architecture decisions (hidden dimensions, attention head counts).
- **Communication in distributed training**: Gradient all-reduce operations can be performed in FP16/BF16 to halve communication bandwidth, a significant optimization for multi-node training.

## Common Misconceptions

- **"Mixed precision means everything is in half precision."** No -- the "mixed" is essential. Optimizer states and master weights are in FP32. Only the forward/backward computations use lower precision.
- **"BF16 is less accurate than FP16."** BF16 has lower mantissa precision, yes, but it has vastly better range. For neural network training, range matters more than precision because overflow/underflow are catastrophic while small rounding errors are tolerable.
- **"Mixed precision training changes the final model quality."** When done correctly, mixed precision produces models that are essentially identical in quality to full FP32 training. The FP32 master copy ensures that long-term training dynamics are preserved.
- **"You need special code for mixed precision."** Modern frameworks (PyTorch AMP, JAX, etc.) handle mixed precision largely automatically with a few configuration flags.

## Connections to Other Concepts

- **Pre-Training**: Mixed precision is essential for making pre-training computationally feasible at scale.
- **Backpropagation**: Gradients are computed in half precision but accumulated and applied in full precision.
- **Adam/AdamW Optimizer**: Optimizer states (moments) are maintained in FP32 as part of the mixed-precision recipe.
- **Distributed Training (Model/Data Parallelism)**: Mixed precision halves communication costs for gradient synchronization.
- **Gradient Clipping**: Applied to gradients after they are cast back to FP32 (or sometimes in half precision before casting).
- **Quantization (Inference)**: Mixed precision training is distinct from inference-time quantization (INT8, INT4), which further reduces precision for deployment.

## Further Reading

- Micikevicius, P., et al. (2018). "Mixed Precision Training" -- The foundational paper that established the loss scaling technique and the mixed precision training recipe.
- Kalamkar, D., et al. (2019). "A Study of BFLOAT16 for Deep Learning Training" -- Demonstrates the effectiveness of BF16 as an alternative to FP16, validating the "same exponent, reduced mantissa" design philosophy.
- NVIDIA (2020). "Training with Mixed Precision" -- A practical guide from NVIDIA on implementing mixed precision training, covering tensor core requirements, loss scaling, and best practices.
