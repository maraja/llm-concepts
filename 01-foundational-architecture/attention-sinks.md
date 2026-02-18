# Attention Sinks

**One-Line Summary**: Attention sinks are the phenomenon where the first few tokens in a sequence accumulate disproportionately large attention scores regardless of their semantic content -- a mathematical artifact of softmax's requirement to produce a valid probability distribution -- and exploiting this property via StreamingLLM enables stable language model inference over millions of tokens with fixed memory.

**Prerequisites**: Self-attention and softmax normalization, KV cache and its memory growth during autoregressive inference, sliding window attention, the concept of perplexity as a measure of model quality, and basic understanding of how transformers process sequences.

## What Is the Attention Sink Phenomenon?

When you examine attention patterns in trained transformers, something odd appears: the very first token in the sequence receives an enormous amount of attention from tokens throughout the entire sequence. This happens regardless of what that first token actually is -- it could be a BOS (beginning-of-sequence) token, a period, the word "the," or any arbitrary content. Across Llama 2, Falcon, Pythia, MPT, and many other architectures, the pattern is consistent.

Why? The answer lies in softmax's constraint. Every attention head must produce weights that sum to 1.0 across all attended positions. But what happens when a token has no particularly relevant prior context to attend to? It has no "none of the above" option. The attention mass must go *somewhere*. In practice, models learn to dump this excess probability mass onto the first token, which acts as a "sink" -- absorbing attention that has nowhere more useful to go.

Think of it like a drainage system. Water (attention mass) must flow somewhere. When there is no meaningful destination, it flows to the lowest point -- the first token, which through training has become the model's learned default drain. The first token's actual content is largely irrelevant; what matters is its consistent position at the start of every training sequence, making it a reliable, predictable dumping ground.

This has a critical practical consequence. If you are performing long inference with a sliding window or limited KV cache and you evict the first token's key-value pairs from memory, perplexity catastrophically spikes to >1000 -- even if you retain thousands of recent tokens. The model breaks not because it lost important semantic information, but because it lost its attention drain.

## How It Works

### The Mathematical Cause

In standard causal self-attention, the attention weights for query at position $t$ are:

$$\alpha_{t,j} = \frac{\exp(q_t \cdot k_j / \sqrt{d})}{\sum_{i=0}^{t} \exp(q_t \cdot k_i / \sqrt{d})}$$

The denominator forces $\sum_j \alpha_{t,j} = 1$. When the query $q_t$ has low dot-product similarity with all available keys, the softmax still must assign probability 1.0 total. Rather than spreading this mass thinly across all tokens (which would create noisy gradients during training), the model learns to concentrate the "nowhere to put this" mass on a consistent location -- the first position.

During training, this concentration emerges because the first token is the only position that appears in every attention computation at every sequence position. It becomes the natural Schelling point for excess attention allocation.

### StreamingLLM: Exploiting Attention Sinks

Xiao et al. (2023) showed that a simple fix enables stable infinite-length inference: keep a small number of "sink tokens" (the first few tokens) permanently in the KV cache alongside a rolling window of recent tokens.

The StreamingLLM KV cache layout:

```
[Sink tokens: 0, 1, 2, 3] [Rolling window: t-W+4, t-W+5, ..., t-1, t]
|--- 4 tokens (fixed) ---| |---------- W-4 tokens (sliding) ----------|
```

The algorithm:
1. **Initialization**: Process the full prompt, keeping all KV pairs.
2. **Eviction**: When the cache exceeds size $W$, evict the oldest tokens *except* the first 4 sink tokens.
3. **Positional encoding**: Re-index positions so that sink tokens occupy positions 0-3 and the rolling window tokens are contiguous from position 4 onward.

With this approach, perplexity remains stable and close to the full-context baseline for sequences tested up to 4 million tokens, using a fixed cache of only a few thousand entries.

### How Many Sink Tokens Are Needed?

Experiments across multiple model families show:
- **0 sink tokens** (pure sliding window): Perplexity explodes after the first tokens exit the window (>1000 perplexity).
- **1 sink token**: Sufficient to maintain stable perplexity in most models.
- **4 sink tokens**: Provides a comfortable safety margin with negligible additional memory cost. This is the default recommendation.
- **More than 4**: No meaningful further improvement.

The fact that even 1 token suffices confirms that the sink phenomenon is about providing a mathematical "drain" rather than preserving any specific semantic information from the beginning of the sequence.

## Why It Matters

1. **Enables streaming inference**: Without attention sinks, any KV cache eviction strategy that removes initial tokens causes catastrophic failure. StreamingLLM makes indefinitely long inference practical with fixed memory.
2. **Explains a fundamental attention pattern**: The attention sink phenomenon provides deep insight into how softmax-based attention actually works in practice -- revealing that a significant portion of the attention mechanism serves a "housekeeping" role rather than processing semantic content.
3. **Informs architecture design**: Understanding sinks has influenced the design of models with dedicated sink tokens, attention-free initial positions, and architectures that handle the excess-attention problem explicitly.
4. **Practical memory savings**: A fixed-size KV cache (e.g., 4 sink + 4092 rolling window tokens) uses constant memory regardless of conversation length, critical for deployment of multi-turn chat systems and streaming applications.
5. **Connects to broader attention research**: The sink phenomenon relates to Vision Transformer registers, differential attention, and other work on understanding and improving attention quality.

## Key Technical Details

- **Perplexity without sinks**: When the first token's KV is evicted from a sliding window cache, perplexity jumps from ~10 to >1000 across all tested models, regardless of window size.
- **Perplexity with sinks**: With 4 sink tokens + 4092 rolling window, perplexity stays within 0.1-0.3 of the full-context baseline, tested up to 4M tokens on Llama 2 7B.
- **Consistency across architectures**: The attention sink pattern and StreamingLLM fix work on Llama 2 (7B, 13B, 70B), Falcon (7B, 40B), Pythia (6.9B, 12B), and MPT (7B, 30B).
- **Attention mass on first token**: In typical attention heads, 20-80% of total attention mass may be directed at the first token for queries with low contextual relevance.
- **No retraining required**: StreamingLLM is a pure inference-time technique. No model modifications or additional training are needed.
- **Position re-indexing**: When using RoPE (Rotary Position Embeddings), the sink tokens must maintain positions 0-3 and the rolling window tokens must be re-indexed to avoid position gaps, which would cause RoPE frequency mismatches and degrade quality.

## Common Misconceptions

- **"The first token receives high attention because it contains important information."** The attention is content-independent. Replacing the first token with random noise and the model still directs significant attention to it. The phenomenon is purely positional and mathematical.
- **"Attention sinks are a bug that should be fixed."** They are an emergent solution to a real problem: softmax must allocate all probability mass somewhere, and the model needs a consistent "nowhere" option. Some architectures now design explicit sink mechanisms, treating this as a feature rather than a bug.
- **"StreamingLLM provides the same quality as full-context attention."** StreamingLLM maintains stable *perplexity* but does not recover information from evicted tokens. It cannot perform tasks that require attending to specific content from the distant past that has been evicted from the window.
- **"This only matters for very long sequences."** The sink phenomenon exists at all sequence lengths -- even short prompts show disproportionate attention on the first token. StreamingLLM is specifically needed for long-context streaming, but the underlying phenomenon is universal.

## Connections to Other Concepts

- **Sliding window attention**: StreamingLLM is built on top of the rolling buffer KV cache from sliding window attention, adding persistent sink tokens to prevent the catastrophic failure that occurs when initial tokens are evicted.
- **KV cache**: Understanding attention sinks is essential for any KV cache management strategy. Eviction policies that remove initial tokens will fail; policies that preserve them (even just 1-4 tokens) succeed.
- **Softmax and attention**: The sink phenomenon is a direct consequence of softmax's normalization constraint. Alternatives like sigmoid attention or attention-free architectures handle this differently.
- **Vision Transformer registers**: Darcet et al. (2023) identified a parallel phenomenon in Vision Transformers, where certain patch tokens accumulate excess attention. Their solution -- adding explicit "register" tokens -- is architecturally analogous to designing dedicated attention sinks.
- **Differential Transformer**: Addresses the same root cause (attention noise from softmax) through a different mechanism: subtracting two attention maps to cancel common-mode noise rather than absorbing it into sink tokens.

## Further Reading

1. **"Efficient Streaming Language Models with Attention Sinks" (Xiao et al., 2023, arXiv:2309.17453)** -- The paper that identified the attention sink phenomenon, proposed StreamingLLM, and demonstrated stable inference over millions of tokens.
2. **"Vision Transformers Need Registers" (Darcet et al., 2023, arXiv:2309.16588)** -- Identifies the parallel attention sink phenomenon in Vision Transformers and proposes explicit register tokens, providing cross-domain validation of the attention sink concept.
3. **"Mistral 7B" (Jiang et al., 2023, arXiv:2310.06825)** -- Introduces the rolling buffer KV cache that StreamingLLM builds upon, demonstrating the practical infrastructure for sliding window inference.
