# Ring Attention

**One-Line Summary**: Ring Attention distributes long sequences across multiple GPUs arranged in a ring topology, overlapping the communication of key-value blocks with attention computation to enable near-linear scaling of context length with the number of devices -- theoretically supporting millions of tokens with less than 5% communication overhead.

**Prerequisites**: Self-attention mechanism and its quadratic memory cost, FlashAttention (blockwise attention with online softmax), distributed training concepts (data parallelism, tensor parallelism), GPU communication primitives (point-to-point send/receive, all-to-all), and KV cache fundamentals.

## What Is Ring Attention?

The context window is one of the most fundamental constraints of transformer models. Even with memory-efficient attention algorithms like FlashAttention, a single GPU can only hold so many tokens before running out of memory. Doubling the context length quadruples the attention computation and doubles the KV memory. At some point, no single device can handle the full sequence.

Ring Attention solves this by distributing the sequence itself across devices. Imagine a group of people sitting in a circle, each holding a different chapter of a book. To understand the full book, each person reads their chapter while simultaneously passing their notes (key-value pairs) to the person on their left. As the notes circulate around the ring, each person computes attention between their chapter (queries) and each set of incoming notes (keys and values). By the time the notes have traveled the full circle, everyone has attended to every chapter. The critical trick is that reading and note-passing happen simultaneously -- while you are computing attention with the current set of notes, the next set is already in transit.

This overlapping of computation and communication is what makes Ring Attention practical. If the attention computation per block takes longer than the time to transfer a block across the interconnect (which it typically does for large enough blocks, since attention is quadratic in block size while communication is linear), the communication is effectively "free" -- hidden entirely behind the computation. The result is that you can process sequences of virtually unlimited length, scaling linearly with the number of GPUs.

## How It Works

### Sequence Partitioning and Ring Topology

Given a sequence of $N$ tokens and $P$ devices, Ring Attention splits the sequence into $P$ contiguous blocks of size $N/P$. Each device $p$ holds:
- Its local query block $Q_p$ (the queries for its portion of the sequence, kept permanently on device)
- Its local key-value block $(K_p, V_p)$ (initially, the KVs for its portion, which will be passed around the ring)

The devices are arranged in a logical ring: device 0 sends to device 1, device 1 to device 2, and so on, with device $P-1$ sending back to device 0 to close the ring.

### The Ring Communication Pattern

The algorithm proceeds in $P$ rounds. In each round $r$, every device simultaneously performs two operations:

1. **Compute**: Device $p$ computes blockwise attention between its local queries $Q_p$ and the currently-held key-value block $(K_{(p-r) \bmod P}, V_{(p-r) \bmod P})$, accumulating the attention output incrementally.
2. **Communicate**: Device $p$ sends its currently-held KV block to device $(p+1) \bmod P$ and receives a new KV block from device $(p-1) \bmod P$.

After $P$ rounds, every device has computed attention between its queries and all $P$ key-value blocks in the sequence, producing the complete attention output for its local portion.

```
Round 0: Device p computes Attn(Q_p, K_p, V_p)         | sends K_p,V_p → next device
Round 1: Device p computes Attn(Q_p, K_{p-1}, V_{p-1})  | sends K_{p-1},V_{p-1} → next device
Round 2: Device p computes Attn(Q_p, K_{p-2}, V_{p-2})  | sends K_{p-2},V_{p-2} → next device
...
Round P-1: Device p computes Attn(Q_p, K_{p+1}, V_{p+1}) | [final round, no more sends needed]
```

### Online Softmax for Incremental Accumulation

A naive implementation would require storing all $N \times N$ attention logits before applying softmax, which would defeat the purpose of distribution. Ring Attention avoids this by using **online softmax** (also called the "safe softmax" or "log-sum-exp trick"), the same technique underlying FlashAttention's blockwise computation. For each incoming KV block, the partial attention output is incrementally merged with the running output:

$$m_{\text{new}} = \max(m_{\text{old}}, m_{\text{block}})$$
$$l_{\text{new}} = e^{m_{\text{old}} - m_{\text{new}}} \cdot l_{\text{old}} + e^{m_{\text{block}} - m_{\text{new}}} \cdot l_{\text{block}}$$
$$O_{\text{new}} = \frac{e^{m_{\text{old}} - m_{\text{new}}} \cdot l_{\text{old}} \cdot O_{\text{old}} + e^{m_{\text{block}} - m_{\text{new}}} \cdot l_{\text{block}} \cdot O_{\text{block}}}{l_{\text{new}}}$$

where $m$ tracks the running maximum logit (for numerical stability) and $l$ tracks the running sum of exponentials. This accumulation is numerically stable and produces results that are *exactly* equivalent to full attention, with zero approximation error.

### Causal Masking and Load Balancing

For causal (autoregressive) models, tokens can only attend to earlier tokens. This means roughly half the attention blocks in the ring are fully masked (queries from early in the sequence cannot attend to keys from later in the sequence) and can be skipped entirely. However, this creates a load imbalance: devices holding early portions of the sequence have less work than devices holding later portions.

**Striped Attention** addresses this by interleaving token assignments across devices rather than using contiguous blocks. Device $p$ holds tokens at positions $\{p, p+P, p+2P, \ldots\}$. This ensures every device has a roughly equal mix of early and late tokens, balancing the causal masking workload evenly across devices.

## Why It Matters

1. **Near-unlimited context length**: Ring Attention scales context length linearly with the number of devices. With 256 GPUs, a model that handles 8K tokens per GPU can process over 2 million tokens.
2. **Minimal overhead**: When block sizes are large enough for computation to dominate communication (which is the typical regime), the overhead is less than 5%. The communication is almost entirely hidden behind the attention computation.
3. **Exact attention**: Unlike sparse or approximate attention methods (Longformer, BigBird, linear attention), Ring Attention computes exact full attention. There is no quality degradation from approximation.
4. **Foundation for frontier models**: The technique (or close variants) underpins models like Gemini 1.5 (1M token context), enabling the longest context windows in production language models.
5. **Composable with other parallelism**: Ring Attention operates on the sequence dimension, orthogonal to tensor parallelism (model dimension) and data parallelism (batch dimension), enabling integration into existing multi-dimensional distributed training frameworks.

## Key Technical Details

- **Memory per device**: $O(N/P \times d)$ for the local sequence block, where $d$ is the model dimension. Plus buffer space for one incoming and one outgoing KV block.
- **Communication volume**: Each device sends and receives $O(N/P \times d)$ data per round, for $P-1$ rounds total. Total communication per device is $O((P-1) \times N/P \times d) \approx O(N \times d)$.
- **Overlap condition**: Communication is fully hidden when $T_{\text{compute}} \geq T_{\text{communicate}}$. For attention, compute scales as $(N/P)^2 \times d$ while communication scales as $(N/P) \times d$. This means the overlap works when $N/P \geq$ some constant, which is easily satisfied for long sequences.
- **Demonstrated scale**: The original paper demonstrated sequences of over 1 million tokens distributed across TPU/GPU clusters, with stable training and near-perfect overlap.
- **Compatibility with FlashAttention**: The blockwise computation within each device naturally uses FlashAttention's tiling strategy, combining inter-device distribution (Ring Attention) with intra-device memory efficiency (FlashAttention).
- **Backward pass**: The ring pattern is applied to the backward pass as well, with gradients flowing in the reverse direction around the ring. The same overlap principle applies.

## Common Misconceptions

- **"Ring Attention is an approximation to full attention."** It computes mathematically exact full attention. The online softmax accumulation is numerically equivalent to computing the full $N \times N$ attention matrix and applying softmax globally. The results are bit-for-bit identical (up to floating point ordering).
- **"Communication overhead makes it impractical."** With proper block sizing, the attention computation per block (quadratic in block size) dominates the communication transfer (linear in block size). For practical long-context configurations, the overlap is near-perfect.
- **"Ring Attention replaces FlashAttention."** They are complementary. FlashAttention optimizes attention computation within a single device using tiling and SRAM management. Ring Attention distributes the sequence across devices. In practice, each device runs FlashAttention for its local block computations.
- **"You need special hardware for the ring topology."** The ring is a logical topology implemented with standard point-to-point send/receive operations that work on any GPU interconnect (NVLink, InfiniBand, PCIe, etc.).

## Connections to Other Concepts

- **FlashAttention**: Provides the blockwise attention primitive and online softmax algorithm that Ring Attention builds upon for its incremental accumulation. Without FlashAttention's insights, the online accumulation would not be practical.
- **Tensor parallelism**: Distributes model weights across devices along the hidden dimension. Ring Attention distributes the sequence. They operate on orthogonal dimensions and can be combined in the same training setup.
- **Sequence parallelism**: A related concept in Megatron-LM that distributes activation memory across the sequence dimension for non-attention operations (LayerNorm, dropout). Ring Attention extends sequence-level distribution to the attention computation itself.
- **Sliding window attention**: An alternative approach to long sequences that restricts attention to a local window, trading global context for efficiency. Ring Attention preserves full global attention over arbitrarily long sequences.
- **Context window extension**: Ring Attention is a key enabling technology for extending context windows beyond what fits on a single device, complementing positional encoding extrapolation techniques like RoPE scaling.

## Further Reading

1. **"Ring Attention with Blockwise Transformers for Near-Infinite Context" (Liu et al., 2023, arXiv:2310.01889)** -- The original Ring Attention paper, presenting the algorithm, overlap analysis, and million-token demonstrations.
2. **"Striped Attention: Faster Ring Attention for Causal Transformers" (Brandon et al., 2023, arXiv:2311.09431)** -- Addresses load imbalance under causal masking by interleaving token assignments across devices, improving throughput for autoregressive models.
3. **"FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness" (Dao et al., 2022, arXiv:2205.14135)** -- The foundation for blockwise attention computation and online softmax that Ring Attention relies upon for its incremental accumulation.
