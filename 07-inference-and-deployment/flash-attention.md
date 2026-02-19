# Flash Attention

**One-Line Summary**: Flash Attention is an IO-aware attention algorithm that restructures the computation to keep data in the GPU's fast on-chip SRAM rather than repeatedly reading and writing to slow high-bandwidth memory (HBM), reducing memory usage from O(N^2) to O(N) and delivering 2-4x wall-clock speedups -- while computing *exact* attention, not an approximation.

**Prerequisites**: Self-attention mechanism (Q, K, V matrices and softmax), GPU architecture basics (what a GPU is and why it has different memory types), Big-O notation for memory and compute complexity, matrix multiplication.

## What Is Flash Attention?

Standard attention has a dirty secret: it is not slow because of the *computation* -- modern GPUs have more than enough arithmetic power. It is slow because of *memory traffic*. The naive implementation writes a massive N x N attention matrix to GPU main memory (HBM), then reads it back to apply softmax, then writes the result again. For long sequences, this memory shuffling dominates the runtime.

![Flash Attention tiling diagram showing how Q, K, V blocks are loaded into SRAM to avoid materializing the full N x N attention matrix in HBM](https://raw.githubusercontent.com/Dao-AILab/flash-attention/main/assets/FlashAttention_banner.jpg)
*Source: [Flash Attention GitHub Repository](https://github.com/Dao-AILab/flash-attention)*


Flash Attention is like reorganizing a kitchen so everything the chef needs is within arm's reach on the counter (SRAM), rather than walking to the pantry (HBM) for every ingredient. The chef cooks the same dish -- the result is identical -- but finishes much faster because the time spent walking is eliminated.

Introduced by Tri Dao et al. in 2022, Flash Attention has become so foundational that it is now integrated into essentially every serious LLM framework and is the default attention implementation in PyTorch.

## How It Works


![GPU memory hierarchy showing the bandwidth gap between SRAM and HBM that Flash Attention exploits](https://gordicaleksa.medium.com/eli5-flash-attention-5c44017022ad)
*See detailed GPU memory hierarchy and tiling diagrams at: [Aleksa Gordic - ELI5 Flash Attention](https://gordicaleksa.medium.com/eli5-flash-attention-5c44017022ad)*

### GPU Memory Hierarchy

Understanding Flash Attention requires understanding the GPU memory hierarchy:

| Memory Level | Size | Bandwidth | Latency |
|-------------|------|-----------|---------|
| SRAM (on-chip) | ~20 MB per SM | ~19 TB/s | ~1 ns |
| HBM (off-chip) | 40-80 GB | ~2-3 TB/s | ~100 ns |

SRAM is roughly 10x faster than HBM, but 1000x smaller. The key insight is that standard attention algorithms are **memory-bandwidth-bound**: the GPU spends most of its time waiting for data transfers, not computing.

### The Problem with Standard Attention

Standard attention computes:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V$$

The naive implementation:

1. Compute S = QK^T. This is an N x N matrix. **Write S to HBM.** (O(N^2) memory write)
2. Read S from HBM. Compute P = softmax(S). **Write P to HBM.** (O(N^2) memory read + write)
3. Read P from HBM. Compute O = PV. **Write O to HBM.** (O(N^2) memory read)

Total HBM accesses: O(N^2). For a sequence length of 8192 with 128-dimensional heads, the attention matrix is 8192 x 8192 = 67 million entries. At FP16, that is 128 MB per head -- far too large for SRAM.

### The Flash Attention Solution: Tiling

Flash Attention never materializes the full N x N attention matrix. Instead, it processes attention in **tiles** (blocks) that fit in SRAM:

1. **Divide Q into blocks** of size B_r (e.g., 64 rows).
2. **Divide K, V into blocks** of size B_c (e.g., 64 columns).
3. For each Q block:
   a. Load the Q block into SRAM.
   b. For each K, V block:
      - Load K and V blocks into SRAM.
      - Compute the partial attention scores: S_block = Q_block * K_block^T / sqrt(d_k).
      - Compute local softmax with running statistics (see below).
      - Accumulate the partial output: O_block += softmax(S_block) * V_block.
   c. Write the final O block to HBM.

The critical detail is that softmax is normally a **global** operation -- you need the maximum and sum across the entire row to normalize. Flash Attention uses the **online softmax trick**:

$$m_{\text{new}} = \max(m_{\text{old}}, \max(S_{\text{block}}))$$
$$\ell_{\text{new}} = e^{m_{\text{old}} - m_{\text{new}}} \cdot \ell_{\text{old}} + \sum_j e^{S_j - m_{\text{new}}}$$
$$O_{\text{new}} = \frac{e^{m_{\text{old}} - m_{\text{new}}} \cdot \ell_{\text{old}} \cdot O_{\text{old}} + e^{S_{\text{block}} - m_{\text{new}}} \cdot V_{\text{block}}}{\ell_{\text{new}}}$$

Running maximum (m) and sum (l) statistics are maintained and corrected as each new block is processed. After all blocks are processed, the result is mathematically identical to standard attention.

### Kernel Fusion

Flash Attention fuses the entire attention computation (QK^T, scaling, masking, softmax, dropout, V multiplication) into a **single GPU kernel**. This eliminates the overhead of launching multiple kernels and the intermediate memory reads/writes between them.

### Memory Reduction

Standard attention: O(N^2) memory for the attention matrix.
Flash Attention: O(N) memory -- only the tile-sized working buffers in SRAM and the output matrix.

This is not just an optimization -- it enables sequence lengths that would otherwise be impossible due to out-of-memory errors.

### Flash Attention 2

Flash Attention 2 (Dao, 2023) improved upon the original with:

- **Better parallelism**: The original parallelized over batch size and number of heads. FA2 also parallelizes over the sequence length dimension, better utilizing GPU SMs.
- **Reduced non-matmul FLOPs**: Reorganized the algorithm to minimize non-matrix-multiplication operations (comparisons, multiplications for softmax rescaling), which are significantly slower on tensor cores.
- **Forward pass**: ~2x speedup over Flash Attention 1, reaching 50-73% of the GPU's theoretical maximum FLOPS (up from 25-40%).
- **Causal masking optimization**: For autoregressive models, skip computation on masked (future) blocks entirely rather than computing and zeroing them.

### Flash Attention 3

Flash Attention 3 (Dao et al., 2024) targets NVIDIA Hopper architecture (H100) with:

*See comparison of standard attention vs Flash Attention memory access patterns at: [Tri Dao's Flash Attention Paper (arXiv:2205.14135)](https://arxiv.org/abs/2205.14135)*


- **Asynchronous operations**: Overlaps data loading (using the Tensor Memory Accelerator, or TMA) with computation using warp specialization. While one set of warps loads the next tile, another set computes on the current tile.
- **FP8 support**: Native FP8 attention computation, leveraging Hopper's FP8 tensor cores for near-2x throughput over FP16.
- **Incoherent processing**: Techniques to manage the reduced precision of FP8 while maintaining accuracy.
- **Block quantization and coherent processing**: Further FP8 accuracy improvements.
- **Achieves 75%+ of H100 theoretical FLOPS**, approaching the hardware ceiling.

## Why It Matters

Flash Attention is not just faster -- it fundamentally changes what is practical:

- **Long context**: Without Flash Attention, processing 128K token sequences would require hundreds of gigabytes for attention matrices alone. Flash Attention makes long-context models possible.
- **Training speed**: 2-4x faster attention translates directly to 15-30% faster end-to-end training (attention is a significant fraction of total compute).
- **Inference efficiency**: Combined with KV cache, Flash Attention ensures that both the prefill phase (compute-bound) and attention operations during decode are as fast as possible.
- **Ubiquity**: Flash Attention is now the default in PyTorch (via `torch.nn.functional.scaled_dot_product_attention`), Hugging Face Transformers, JAX, and every major serving framework. It is no longer an optimization you enable -- it is the baseline.

## Key Technical Details

- **Exact computation**: Flash Attention computes *exact* attention. It is not an approximation like sparse attention or linear attention. The output is bitwise identical (up to floating-point reordering) to standard attention.
- **Hardware specificity**: Block sizes and tile dimensions are tuned to specific GPU architectures. What is optimal for A100 differs from H100. The implementation auto-tunes or selects based on the GPU.
- **Backward pass**: Flash Attention also speeds up the backward pass by recomputing attention scores from Q, K (stored in HBM) rather than saving the O(N^2) attention matrix for backpropagation. This trades a small amount of extra compute for massive memory savings.
- **Dropout handling**: Generates dropout masks on-the-fly in SRAM rather than storing an N x N dropout mask in HBM.
- **Sliding window attention**: Flash Attention natively supports sliding window (local) attention patterns, as used in Mistral, by skipping blocks that are outside the attention window.

## Common Misconceptions

- **"Flash Attention is an approximate attention method."** This is the most common and most damaging misconception. Flash Attention computes *exact* standard attention. It merely reorganizes the order of operations to minimize memory traffic. The output is the same.
- **"Flash Attention only helps with long sequences."** While the benefits grow with sequence length (the O(N^2) vs. O(N) memory reduction matters more for larger N), Flash Attention also speeds up short-sequence attention by 1.5-2x through kernel fusion alone.
- **"Flash Attention replaces KV cache."** They solve different problems. KV cache avoids redundant computation across time steps. Flash Attention speeds up each individual attention computation. They are complementary.
- **"You need to install Flash Attention separately."** As of PyTorch 2.0+, the `scaled_dot_product_attention` function automatically uses a Flash Attention-style kernel when available. Explicit installation of the `flash-attn` package is still needed for the most optimized version.
- **"All attention patterns benefit equally."** Flash Attention is most impactful for dense (full) attention. For very sparse attention patterns, other approaches may be more efficient.

## Connections to Other Concepts

- **KV Cache**: Flash Attention optimizes the attention computation itself; KV cache avoids recomputing K and V. During the decode phase, Flash Attention handles the single-query-against-full-cache operation efficiently.
- **Quantization**: Flash Attention 3's FP8 support connects directly to the quantization story, enabling reduced-precision attention computation on Hopper GPUs.
- **Model Serving Frameworks**: Every major serving framework (vLLM, TGI, TensorRT-LLM) integrates Flash Attention as a non-negotiable baseline optimization.
- **Throughput vs. Latency**: Flash Attention's speedup benefits both throughput (more tokens per second in batch) and latency (faster per-request attention).
- **Speculative Decoding**: During the verification step, the target model processes multiple tokens at once -- Flash Attention ensures this batched attention is fast.

## Further Reading

1. **"FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness"** (Dao et al., 2022) -- The original paper that introduced the IO-aware tiling approach and demonstrated its dramatic impact.
2. **"FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning"** (Dao, 2023) -- The follow-up with improved parallelism and non-matmul FLOP reduction, achieving 50-73% of theoretical GPU throughput.
3. **"FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision"** (Shah et al., 2024) -- The Hopper-optimized version with FP8 support and asynchronous warp-specialized execution.
