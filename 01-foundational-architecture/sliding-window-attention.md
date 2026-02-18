# Sliding Window Attention

**One-Line Summary**: Sliding window attention restricts each token's attention to a fixed-size local window of $W$ neighboring tokens, reducing the quadratic memory cost of full attention to linear while preserving long-range information flow through layer stacking -- where each additional layer extends the effective receptive field by $W$ tokens.

**Prerequisites**: Self-attention mechanism, multi-head attention, the quadratic cost problem of standard attention ($O(n^2)$), KV cache for autoregressive inference, and an understanding of receptive fields from convolutional neural networks.

## What Is Sliding Window Attention?

Standard self-attention lets every token attend to every other token in the sequence. This is powerful but expensive: for a sequence of length $n$, attention requires $O(n^2)$ memory and compute. Double the sequence length, and you quadruple the cost. At 128K tokens, full attention becomes prohibitively expensive for most hardware.

Sliding window attention takes a pragmatic approach borrowed from convolutional neural networks: each token only attends to its $W$ nearest neighbors. Think of it like reading a book through a magnifying glass that shows exactly $W$ words at a time. Each word can only "see" the words within its glass, but as you slide the glass across the page, every word gets its turn in context. The key realization is that in a deep transformer, these local windows compound -- just as stacking convolutional layers builds a larger receptive field, stacking attention layers with window size $W$ means information can propagate $L \times W$ tokens across $L$ layers.

This insight was deployed to great effect in Mistral 7B, which used $W = 4096$ with 32 layers, creating an effective receptive field of $32 \times 4096 = 131{,}072$ tokens while keeping per-layer memory costs fixed. The model outperformed the significantly larger Llama 2 13B while running roughly twice as fast at inference.

## How It Works

### The Attention Mask

In standard self-attention, token $i$ attends to all tokens $j \in [0, i]$ (for causal/autoregressive models). With sliding window attention, token $i$ attends only to tokens $j \in [\max(0, i - W + 1), i]$, where $W$ is the window size. The attention computation becomes:

$$\text{Attention}(Q_i, K, V) = \text{softmax}\left(\frac{Q_i K_{[i-W+1:i]}^T}{\sqrt{d_k}}\right) V_{[i-W+1:i]}$$

This is implemented via a banded attention mask -- a diagonal band of width $W$ in the $n \times n$ attention matrix. Tokens outside the window receive $-\infty$ masking before the softmax, exactly like causal masking but additionally zeroing out positions that fall before the window's start.

The memory reduction is immediate: instead of storing an $n \times n$ attention matrix, you store an $n \times W$ matrix. For $n = 32{,}768$ and $W = 4{,}096$, this is an 8x reduction in attention memory alone. The compute reduction follows the same factor, since you evaluate $n \times W$ dot products instead of $n \times n$.

### Effective Receptive Field Through Layer Stacking

The power of sliding window attention comes from stacking. At layer 1, token $t$ can access tokens in $[t - W + 1, t]$. But the tokens at position $t - W + 1$ themselves attended to tokens in $[t - 2W + 2, t - W + 1]$ at the previous layer. So at layer 2, token $t$ has *indirect* access to tokens as far back as $t - 2W + 2$.

Generalizing, at layer $l$, the effective receptive field extends back by $l \times (W - 1) + 1$ positions. For Mistral 7B with 32 layers and $W = 4096$:

$$\text{Effective receptive field} = 32 \times 4095 + 1 = 131{,}041 \text{ tokens}$$

This means the last layer of the model can, in principle, be influenced by information from over 131K tokens ago, even though each individual attention layer only looks at 4,096 tokens. The information "flows" through intermediate tokens across successive layers, analogous to how information propagates through the layers of a deep CNN.

### Rolling Buffer KV Cache

The KV cache is a critical optimization for autoregressive inference, storing previously computed key-value pairs to avoid recomputation. In full attention, the KV cache grows linearly with sequence length -- at token 100K, you are storing 100K key-value pairs per layer.

Sliding window attention enables a **rolling buffer** (circular buffer) of fixed size $W$. Since token $i$ only attends to the most recent $W$ tokens, key-value pairs older than $W$ positions can be safely evicted. The cache position is computed as $i \mod W$, creating a circular overwrite pattern:

```
Cache position = token_position % W
# Token 0 → slot 0, Token 1 → slot 1, ...
# Token W → slot 0 (overwrites token 0), Token W+1 → slot 1, ...
```

This bounds KV cache memory at exactly $W \times 2 \times d_{\text{head}} \times n_{\text{heads}} \times n_{\text{layers}}$ regardless of sequence length. For Mistral 7B ($W = 4096$, $d_{\text{head}} = 128$, 32 layers, 8 KV heads in GQA), this is a fixed ~2GB, whether you have generated 100 tokens or 1 million.

### Pre-fill Chunking

For long input prompts, the pre-fill phase (processing the entire prompt before generating) can still be memory-intensive. Sliding window attention enables chunked pre-fill: the prompt is processed in chunks of size $W$, with each chunk attending to itself and the KV cache from the previous chunk. This bounds peak memory during pre-fill to the same $O(W^2)$ attention cost per chunk per layer, rather than requiring the full $O(n^2)$ for the entire prompt in a single pass. For a 32K prompt with $W = 4096$, this means 8 chunks processed sequentially, each using 4K x 4K attention rather than one 32K x 32K attention computation.

## Why It Matters

1. **Linear memory scaling**: Memory costs grow as $O(n \times W)$ instead of $O(n^2)$, enabling processing of much longer sequences on the same hardware without any architectural change to the attention mechanism itself.
2. **Fixed-size inference cache**: The rolling buffer KV cache uses constant memory regardless of how many tokens have been generated, making sliding window models uniquely suited for streaming, multi-turn conversations, and long-running inference.
3. **Effective long-range modeling**: Through layer stacking, the effective receptive field is $L \times W$, which for Mistral 7B reaches ~131K tokens -- far beyond what most applications require at any single moment.
4. **Superior quality-efficiency tradeoff**: Mistral 7B achieved 60.1% on MMLU versus Llama 2 13B's 55.4%, with approximately 2x faster inference, demonstrating that full attention across all positions is often wasteful.
5. **Composability**: Sliding window attention can be combined with grouped-query attention (as in Mistral 7B), FlashAttention for IO-efficient computation, and attention sinks for infinite-length streaming.

## Key Technical Details

- **Window size $W = 4{,}096$** in Mistral 7B, chosen to balance local context quality with memory efficiency. Larger windows improve quality but increase per-layer cost.
- **Mistral 7B benchmarks**: 60.1% MMLU, 52.2% HellaSwag, 75.2% ARC-Challenge -- outperforming Llama 2 13B (a model nearly twice its parameter count) on most benchmarks.
- **Information propagation**: At layer $l$, a token has access to information from up to $l \times (W-1) + 1$ tokens away. At the final layer of a 32-layer model with $W = 4096$, this exceeds 131K tokens.
- **Attention pattern sparsity**: Empirical analysis of full-attention models shows that most attention mass is concentrated in local windows anyway. Sliding window attention formalizes this observation as an architectural constraint.
- **Inference speedup**: Beyond memory savings, sliding window attention reduces compute from $O(n^2 d)$ to $O(nWd)$ per layer, which translates to measurable wall-clock speedups for long sequences.
- **Hybrid approaches**: Mixtral and some later models alternate between sliding window and full attention layers, capturing both local detail and global context. This hybrid pattern preserves some global attention capacity while keeping the average cost low.

## Common Misconceptions

- **"Sliding window attention can't handle long-range dependencies."** Through layer stacking, information propagates across the full effective receptive field. The model learns to relay important information through intermediate positions across layers, similar to how deep CNNs build global understanding from local filters.
- **"The window size must match the context length."** The window size is typically much smaller than the supported context length. Mistral 7B processes 32K+ tokens with a 4K window, relying on stacked layers for long-range propagation.
- **"This is the same as Longformer or BigBird."** Longformer combines local windows with global attention tokens that attend to every position. BigBird adds random attention connections on top of local windows. Sliding window attention as used in Mistral is purely local, relying entirely on layer stacking rather than global tokens or random connections for long-range information flow.
- **"You lose the first tokens when using the rolling buffer."** This is true and intentional -- the rolling buffer discards KV entries older than $W$ positions. For applications needing persistent access to initial tokens, techniques like attention sinks (keeping the first 4 tokens permanently) can be combined with the rolling buffer.

## Connections to Other Concepts

- **Full self-attention**: Sliding window attention is a strict subset of full attention -- it applies the same dot-product attention mechanism but restricts the set of keys and values each query can access to a local neighborhood.
- **KV cache**: The rolling buffer optimization transforms the standard growing KV cache into a fixed-size circular buffer, fundamentally changing the memory profile of autoregressive inference.
- **Sparse attention**: Sliding window is one specific pattern of sparse attention. Others include strided patterns (Sparse Transformers), local+global (Longformer), and random (BigBird). Each makes different tradeoffs between locality, global coverage, and complexity.
- **Grouped-query attention (GQA)**: Mistral 7B combines sliding window with GQA (8 KV heads vs. 32 query heads), stacking two independent efficiency optimizations for compounding memory savings.
- **Attention sinks**: StreamingLLM builds on the rolling buffer idea by retaining a few initial "sink" tokens alongside the sliding window, preventing perplexity degradation on very long sequences where the initial tokens' KV entries would otherwise be evicted.

## Further Reading

1. **"Mistral 7B" (Jiang et al., 2023, arXiv:2310.06825)** -- Introduces the practical deployment of sliding window attention with rolling buffer KV cache, demonstrating state-of-the-art efficiency-performance tradeoffs at 7B scale.
2. **"Longformer: The Long-Document Transformer" (Beltagy et al., 2020, arXiv:2004.05150)** -- Combines sliding window attention with global attention tokens for document-level tasks, providing an important alternative design and comparison point.
3. **"Generating Long Sequences with Sparse Transformers" (Child et al., 2019, arXiv:1904.10509)** -- Explores various sparse attention patterns including local windows and strided attention, laying theoretical groundwork for efficient attention.
