# Ring Attention

**One-Line Summary**: Ring Attention distributes long sequences across multiple GPUs arranged in a ring topology, overlapping the communication of key-value blocks with attention computation to enable near-linear scaling of context length with the number of devices -- supporting millions of tokens with less than 5% communication overhead.

**Prerequisites**: Self-attention and its quadratic memory cost, FlashAttention (blockwise attention with online softmax), distributed training concepts (data parallelism, tensor parallelism), GPU communication primitives (point-to-point send/receive), and KV cache fundamentals.

## What Is Ring Attention?

The context window is one of the most fundamental constraints of transformer models. Even with FlashAttention, a single GPU can only hold so many tokens before running out of memory. Doubling context length quadruples attention computation and doubles KV memory. At some point, no single device can handle the full sequence.

*Recommended visual: Ring topology diagram showing KV blocks circulating between devices while each device computes attention with its local queries -- see [Liu et al., "Ring Attention with Blockwise Transformers for Near-Infinite Context" (2023)](https://arxiv.org/abs/2310.01889), Figure 1*


Ring Attention solves this by distributing the sequence itself across devices. Imagine a group of people sitting in a circle, each holding a different chapter of a book. To understand the full book, each person reads their chapter while simultaneously passing their notes (key-value pairs) to the person on their left.

As the notes circulate around the ring, each person computes attention between their chapter (queries) and each set of incoming notes. By the time the notes have traveled the full circle, everyone has attended to every chapter.

The critical trick: computation and communication happen simultaneously. While you compute attention with the current KV block, the next block is already in transit. Since attention computation (quadratic in block size) typically takes longer than data transfer (linear in block size), the communication is effectively free.

## How It Works


*Recommended visual: Computation-communication overlap timeline showing how attention computation on the current KV block hides the transfer latency of the next block -- see [Liu et al., "Ring Attention" (2023)](https://arxiv.org/abs/2310.01889), Figure 2*

### Sequence Partitioning and Ring Topology

Given $N$ tokens and $P$ devices, Ring Attention splits the sequence into $P$ contiguous blocks of $N/P$ tokens. Each device $p$ holds:

- Its local query block $Q_p$ (kept permanently on device)
- Its local key-value block $(K_p, V_p)$ (initially local, will be passed around the ring)

Devices form a logical ring: device 0 sends to device 1, device 1 to device 2, ..., device $P-1$ back to device 0.

### The Ring Communication Pattern

The algorithm proceeds in $P$ rounds. In each round $r$, every device simultaneously:

1. **Computes**: Blockwise attention between $Q_p$ and the currently-held KV block, accumulating the output incrementally.
2. **Communicates**: Sends the current KV block to the next device; receives a new KV block from the previous device.

```
Round 0: Device p computes Attn(Q_p, K_p, V_p)         | sends K_p,V_p → next
Round 1: Device p computes Attn(Q_p, K_{p-1}, V_{p-1})  | sends K_{p-1},V_{p-1} → next
...
Round P-1: Device p computes Attn(Q_p, K_{p+1}, V_{p+1}) | [final round]
```

After $P$ rounds, every device has computed attention between its queries and all KV blocks.

### Online Softmax for Incremental Accumulation

A naive approach would store all $N \times N$ attention logits before applying softmax. Ring Attention avoids this using **online softmax** (from FlashAttention):

$$m_{\text{new}} = \max(m_{\text{old}}, m_{\text{block}})$$
$$l_{\text{new}} = e^{m_{\text{old}} - m_{\text{new}}} \cdot l_{\text{old}} + e^{m_{\text{block}} - m_{\text{new}}} \cdot l_{\text{block}}$$
$$O_{\text{new}} = \frac{e^{m_{\text{old}} - m_{\text{new}}} \cdot l_{\text{old}} \cdot O_{\text{old}} + e^{m_{\text{block}} - m_{\text{new}}} \cdot l_{\text{block}} \cdot O_{\text{block}}}{l_{\text{new}}}$$

Here $m$ tracks the running maximum logit and $l$ the running sum of exponentials. This is numerically stable and *exactly* equivalent to full attention -- zero approximation error.

### Causal Masking and Load Balancing

For causal models, tokens only attend to earlier tokens. Roughly half the attention blocks are fully masked and can be skipped, but this creates load imbalance: devices with early sequence portions have less work.

**Striped Attention** fixes this by interleaving token assignments: device $p$ holds tokens at positions $\{p, p+P, p+2P, \ldots\}$. Every device gets a mix of early and late tokens, balancing the workload.

### Backward Pass

The ring pattern applies to the backward pass as well. Gradients flow in the reverse direction around the ring, with the same computation-communication overlap. Each device computes local gradients with respect to its queries while circulating the KV blocks and their gradients.

*Recommended visual: Striped attention token assignment pattern for causal masking load balance, contrasting block-contiguous vs interleaved partitioning -- see [Brandon et al., "Striped Attention" (2023)](https://arxiv.org/abs/2311.09431), Figure 1*


## Why It Matters

1. **Near-unlimited context**: Scales context linearly with devices. 256 GPUs with 8K tokens each = 2M+ token context.
2. **Minimal overhead**: Less than 5% when computation dominates communication (typical for large blocks).
3. **Exact attention**: No approximation, no quality loss -- mathematically identical to full attention.
4. **Foundation for frontier models**: Variants underpin Gemini 1.5's 1M token context and other long-context systems.
5. **Composable**: Orthogonal to tensor parallelism and data parallelism, enabling multi-dimensional distribution.

## Key Technical Details

- **Memory per device**: $O(N/P \times d)$ for the local block, plus double-buffer for KV transfer.
- **Communication volume**: Each device transfers $O(N \times d)$ total across all $P-1$ rounds.
- **Overlap condition**: Communication is hidden when $(N/P)^2 \times d > (N/P) \times d / \text{bandwidth}$, easily satisfied for long sequences.
- **Demonstrated scale**: Millions of tokens across GPU/TPU clusters with stable training.
- **FlashAttention compatibility**: Each device uses FlashAttention's tiling for local block computation.
- **Multi-head amortization**: Ring communication is shared across attention heads, amortizing transfer costs.
- **Practical deployment**: Believed to underpin context extension in Gemini 1.5 and similar frontier systems.
- **Bandwidth requirements**: Modern NVLink (900 GB/s) or InfiniBand (400 Gb/s) provide sufficient bandwidth for most configurations.

## Common Misconceptions

- **"Ring Attention is an approximation."** It computes exact full attention via online softmax. Results are mathematically identical to standard attention.
- **"Communication overhead makes it impractical."** Attention computation (quadratic in block size) dominates communication (linear), making overlap near-perfect.
- **"Ring Attention replaces FlashAttention."** They are complementary: FlashAttention handles within-device tiling, Ring Attention handles across-device distribution.
- **"You need special ring hardware."** The ring is logical, implemented with standard point-to-point operations on any interconnect.

## Connections to Other Concepts

- **FlashAttention**: Provides the blockwise computation and online softmax that Ring Attention uses for incremental accumulation.
- **Tensor parallelism**: Distributes model weights across devices. Ring Attention distributes the sequence. They are orthogonal.
- **Sequence parallelism**: Megatron-LM distributes activations across the sequence for non-attention ops. Ring Attention extends this to attention itself.
- **Sliding window attention**: An alternative for long sequences via locality. Ring Attention preserves full global attention.
- **Context window extension**: Ring Attention is a key enabler for extending context beyond single-device limits.

## Further Reading

1. **"Ring Attention with Blockwise Transformers for Near-Infinite Context" (Liu et al., 2023, arXiv:2310.01889)** -- The original paper presenting the algorithm, overlap analysis, and million-token demonstrations.
2. **"Striped Attention: Faster Ring Attention for Causal Transformers" (Brandon et al., 2023, arXiv:2311.09431)** -- Addresses causal masking load imbalance via interleaved token assignment.
3. **"FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness" (Dao et al., 2022, arXiv:2205.14135)** -- The foundation for blockwise computation and online softmax.
