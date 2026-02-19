# Sliding Window Attention

**One-Line Summary**: Sliding window attention restricts each token's attention to a fixed-size local window of $W$ neighboring tokens, reducing the quadratic memory cost of full attention to linear while preserving long-range information flow through layer stacking -- where each additional layer extends the effective receptive field by $W$ tokens.

**Prerequisites**: Self-attention mechanism, multi-head attention, the quadratic cost problem of standard attention ($O(n^2)$), KV cache for autoregressive inference, and an understanding of receptive fields from convolutional neural networks.

## What Is Sliding Window Attention?

Standard self-attention lets every token attend to every other token in the sequence. This is powerful but expensive: for a sequence of length $n$, attention requires $O(n^2)$ memory and compute. Double the sequence length, and you quadruple the cost.

Sliding window attention takes a pragmatic approach borrowed from convolutional neural networks: each token only attends to its $W$ nearest neighbors. Think of it like reading a book through a magnifying glass that shows exactly $W$ words at a time.

The key realization is that in a deep transformer, these local windows compound. Just as stacking convolutional layers builds a larger receptive field, stacking attention layers with window size $W$ means information can propagate $L \times W$ tokens across $L$ layers.

Mistral 7B used $W = 4096$ with 32 layers, creating an effective receptive field of $32 \times 4096 = 131{,}072$ tokens while keeping per-layer memory costs fixed. The model outperformed the significantly larger Llama 2 13B while running roughly twice as fast at inference.

## How It Works

### The Attention Mask

In standard causal self-attention, token $i$ attends to all tokens $j \in [0, i]$. With sliding window attention, token $i$ attends only to tokens $j \in [\max(0, i - W + 1), i]$. The attention computation becomes:

$$\text{Attention}(Q_i, K, V) = \text{softmax}\left(\frac{Q_i K_{[i-W+1:i]}^T}{\sqrt{d_k}}\right) V_{[i-W+1:i]}$$

This is implemented via a banded attention mask -- a diagonal band of width $W$ in the $n \times n$ attention matrix. Tokens outside the window receive $-\infty$ masking before the softmax.

The memory reduction is immediate: instead of an $n \times n$ attention matrix, you store $n \times W$. For $n = 32{,}768$ and $W = 4{,}096$, this is an 8x reduction. Compute scales identically.

### Effective Receptive Field Through Layer Stacking

At layer 1, token $t$ can access tokens in $[t - W + 1, t]$. But the tokens at position $t - W + 1$ themselves attended to tokens in $[t - 2W + 2, t - W + 1]$ at the previous layer. At layer 2, token $t$ has indirect access to tokens as far back as $t - 2W + 2$.

Generalizing, at layer $l$, the effective receptive field extends back by $l \times (W - 1) + 1$ positions:

$$\text{Effective receptive field} = 32 \times 4095 + 1 = 131{,}041 \text{ tokens (for Mistral 7B)}$$

Information "flows" through intermediate tokens across successive layers, analogous to how information propagates through the layers of a deep CNN.

### Rolling Buffer KV Cache

The KV cache stores previously computed key-value pairs to avoid recomputation during autoregressive inference. In full attention, this cache grows linearly with sequence length -- at token 100K, you store 100K KV pairs per layer.

Sliding window attention enables a **rolling buffer** (circular buffer) of fixed size $W$:

```
Cache position = token_position % W
# Token 0 → slot 0, Token 1 → slot 1, ...
# Token W → slot 0 (overwrites token 0), Token W+1 → slot 1, ...
```

This bounds KV cache memory at exactly $W \times 2 \times d_{\text{head}} \times n_{\text{heads}} \times n_{\text{layers}}$ regardless of sequence length. For Mistral 7B ($W = 4096$, $d_{\text{head}} = 128$, 32 layers, 8 KV heads in GQA), this is a fixed ~2GB whether you have generated 100 tokens or 1 million.

### Pre-fill Chunking

For long input prompts, the pre-fill phase can be memory-intensive. Sliding window attention enables chunked pre-fill: the prompt is processed in chunks of size $W$, with each chunk attending to itself and the KV cache from the previous chunk.

For a 32K prompt with $W = 4096$, this means 8 chunks of 4K each, rather than one 32K x 32K attention computation.

### Comparison with Other Efficient Attention Methods

- **Longformer**: Local windows + designated global attention tokens. More complex masking, additional design decisions about which tokens are global.
- **BigBird**: Local windows + random connections. Theoretical connectivity guarantees but harder to implement efficiently.
- **Sparse Transformers**: Strided patterns where every $k$-th token attends globally. Effective but requires careful stride tuning.
- **Linear Attention**: $O(n)$ via kernel approximation, but typically with quality degradation on language tasks.

Sliding window's advantage is simplicity: one hyperparameter ($W$), straightforward masking, and natural compatibility with existing FlashAttention kernels.

## Why It Matters

1. **Linear memory scaling**: Memory grows as $O(n \times W)$ instead of $O(n^2)$, enabling much longer sequences on the same hardware.
2. **Fixed-size inference cache**: Constant memory regardless of tokens generated -- ideal for streaming and multi-turn conversations.
3. **Effective long-range modeling**: Effective receptive field of $L \times W$ reaches ~131K tokens for Mistral 7B.
4. **Superior quality-efficiency tradeoff**: Mistral 7B (60.1% MMLU) outperforms Llama 2 13B (55.4%) at 2x inference speed.
5. **Composability**: Combines with GQA, FlashAttention, and attention sinks for compounding efficiency gains.

## Key Technical Details

- **Window size**: $W = 4{,}096$ in Mistral 7B. Larger windows improve quality but increase per-layer cost.
- **Mistral 7B benchmarks**: 60.1% MMLU, 52.2% HellaSwag, 75.2% ARC-Challenge.
- **Attention pattern sparsity**: Full-attention models concentrate most mass in local windows anyway -- sliding window formalizes this.
- **Inference speedup**: Reduces per-layer compute from $O(n^2 d)$ to $O(nWd)$, ~2x wall-clock speedup for long sequences.
- **Hybrid approaches**: Mixtral alternates sliding window and full attention layers for both local detail and global context.
- **FlashAttention compatibility**: The banded mask pattern integrates naturally with FlashAttention's tiling.
- **Position encoding**: RoPE works naturally with sliding window, as relative positions within the window are preserved.

## Common Misconceptions

- **"Sliding window attention can't handle long-range dependencies."** Through layer stacking, information propagates across the full effective receptive field. The model learns to relay important information through intermediate tokens.
- **"The window size must match the context length."** Mistral 7B processes 32K+ tokens with a 4K window, relying on stacked layers for long-range propagation.
- **"This is the same as Longformer or BigBird."** Those combine local windows with global or random attention. Sliding window (as in Mistral) is purely local, relying entirely on layer stacking.
- **"You lose the first tokens with the rolling buffer."** True and intentional. For applications needing persistent access to initial tokens, attention sinks can be combined with the rolling buffer.

## Connections to Other Concepts

- **Full self-attention**: Sliding window is a strict subset -- same mechanism, restricted key-value set.
- **KV cache**: The rolling buffer transforms a growing cache into a fixed-size circular buffer.
- **Sparse attention**: Sliding window is one pattern; others include strided, local+global, and random.
- **Grouped-query attention**: Mistral 7B combines sliding window with GQA for compounding memory savings.
- **Attention sinks**: StreamingLLM adds persistent sink tokens to the rolling buffer, preventing perplexity degradation.

## Diagrams and Visualizations

*Recommended visual: Sliding window attention pattern showing the local window of W tokens per layer, with the effective receptive field growing across layers — see [Mistral 7B Paper (arXiv:2310.06825)](https://arxiv.org/abs/2310.06825)*

*Recommended visual: Rolling buffer KV cache diagram showing fixed-size cache with position modulo addressing — see [Mistral AI Documentation](https://docs.mistral.ai/)*

## Further Reading

1. **"Mistral 7B" (Jiang et al., 2023, arXiv:2310.06825)** -- Introduces sliding window attention with rolling buffer KV cache in practice.
2. **"Longformer: The Long-Document Transformer" (Beltagy et al., 2020, arXiv:2004.05150)** -- Combines sliding windows with global attention tokens for document-level tasks.
3. **"Generating Long Sequences with Sparse Transformers" (Child et al., 2019, arXiv:1904.10509)** -- Explores sparse attention patterns including local windows and strided attention.
