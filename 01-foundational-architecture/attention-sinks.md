# Attention Sinks

**One-Line Summary**: Attention sinks are the phenomenon where the first few tokens in a sequence accumulate disproportionately large attention scores regardless of their semantic content -- a mathematical artifact of softmax's requirement to produce a valid probability distribution -- and exploiting this property via StreamingLLM enables stable language model inference over millions of tokens with fixed memory.

**Prerequisites**: Self-attention and softmax normalization, KV cache and its memory growth during autoregressive inference, sliding window attention and rolling buffer caches, the concept of perplexity as a measure of language model quality, and basic understanding of how transformers process sequences position by position.

## What Is the Attention Sink Phenomenon?

When you examine the attention patterns of trained transformers, something surprising appears: the very first token in the sequence receives an enormous amount of attention from tokens throughout the entire sequence. This happens regardless of what that first token actually is -- it could be a BOS (beginning-of-sequence) special token, a period, the word "the," or any arbitrary content. Across Llama 2, Falcon, Pythia, MPT, and many other model families, the pattern is remarkably consistent.

![Attention heatmap showing disproportionate attention mass concentrated on the first few tokens (attention sinks) regardless of their semantic content, with the sink pattern visible across multiple layers](https://github.com/mit-han-lab/streaming-llm/raw/main/figures/attention_sink.png)
*Source: [StreamingLLM -- MIT Han Lab GitHub Repository](https://github.com/mit-han-lab/streaming-llm)*


Why does this happen? The answer lies in softmax's fundamental constraint. Every attention head must produce weights that sum to exactly 1.0 across all attended positions. But what happens when a query token has no particularly relevant prior context to attend to? There is no "none of the above" option in the softmax distribution. The attention mass *must* go somewhere. In practice, models learn during training to dump this excess probability mass onto the first token, which acts as a "sink" -- absorbing attention that has nowhere more useful to go.

Think of it like a drainage system. Water (attention mass) must flow somewhere because the system is closed (softmax sums to 1). When there is no meaningful destination for the water, it flows to the lowest point -- the first token position, which through millions of training steps has become the model's learned default drain. The first token's actual content is largely irrelevant to this function; what matters is its consistent position at the beginning of every training sequence, making it a reliable, predictable dumping ground that all attention heads can coordinate around.

This phenomenon has a critical practical consequence for long-context inference. If you use a sliding window or limited KV cache and evict the first token's key-value pairs from memory, perplexity catastrophically spikes to values exceeding 1000 -- even if you retain thousands of recent tokens with meaningful content. The model breaks not because it lost important semantic information, but because it lost its attention drain.

## How It Works


![StreamingLLM KV cache layout showing fixed sink tokens at the beginning plus a rolling window of recent tokens, with evicted middle tokens creating a gap that does not degrade perplexity](https://github.com/mit-han-lab/streaming-llm/raw/main/figures/streaming_llm.png)
*Source: [StreamingLLM -- MIT Han Lab GitHub Repository](https://github.com/mit-han-lab/streaming-llm)*

### The Mathematical Cause

In standard causal self-attention, the attention weights for a query at position $t$ attending to all previous positions are:

$$\alpha_{t,j} = \frac{\exp(q_t \cdot k_j / \sqrt{d})}{\sum_{i=0}^{t} \exp(q_t \cdot k_i / \sqrt{d})}$$

The denominator forces $\sum_j \alpha_{t,j} = 1$. When the query $q_t$ has low dot-product similarity with all available keys -- meaning no previous token is particularly relevant -- the softmax still must distribute probability 1.0 across all positions. Rather than spreading this mass thinly and uniformly across all tokens (which would create noisy gradients during training and blur the attention pattern), the model learns to concentrate the "nowhere useful to put this" mass on a single, consistent location: the first position.

This concentration emerges during training because the first token is the only position that appears in the attention window of every single query at every sequence position. It is the universal constant across all attention computations in a causal model. Through gradient descent, the keys at position 0 learn to act as a "soft no-op" -- they accept attention mass without significantly influencing the output (their values are effectively learned to be near-neutral for this purpose). This makes the first position a natural Schelling point for excess attention allocation.

### StreamingLLM: Exploiting Attention Sinks for Infinite Inference

Xiao et al. (2023) showed that understanding attention sinks leads to a simple but powerful fix for infinite-length inference: keep a small number of "sink tokens" (the first few tokens) permanently in the KV cache alongside a rolling window of recent tokens. The KV cache layout becomes:

```
[Sink tokens: 0, 1, 2, 3] [Gap: tokens evicted] [Rolling window: t-W+4, ..., t-1, t]
|--- 4 tokens (fixed) ---|                       |---------- W-4 tokens (sliding) --------|
```

The StreamingLLM algorithm:
1. **Initialization**: Process the full prompt normally, populating the KV cache for all positions.
2. **Eviction policy**: When the cache exceeds its maximum size $W$, evict the oldest tokens *except* the first 4 sink tokens. The sink tokens are never evicted.
3. **Positional re-indexing**: Critically, the positions assigned to tokens in the cache must be re-indexed so that sink tokens occupy positions 0-3 and the rolling window tokens are assigned contiguous positions starting from position 4. This avoids gaps in positional encodings (especially important for RoPE), which would cause frequency mismatches and degrade generation quality.

With this approach, perplexity remains stable and close to the full-context baseline for sequences tested up to 4 million tokens, using a fixed cache of only a few thousand entries.

### How Many Sink Tokens Are Needed?

Systematic experiments across multiple model families reveal a clear pattern:
- **0 sink tokens** (pure sliding window with eviction): Perplexity explodes catastrophically to >1000 as soon as the first tokens exit the window.
- **1 sink token**: Sufficient to maintain stable perplexity in most tested models. The single sink provides the minimum viable attention drain.
- **4 sink tokens**: Provides a comfortable safety margin with negligible additional memory cost (4 KV entries out of thousands). This is the recommended default.
- **More than 4**: No meaningful further improvement in perplexity stability.

The fact that even a single token suffices confirms that the phenomenon is about providing a mathematical "drain" for excess attention mass rather than preserving any specific semantic content from the beginning of the sequence.

## Why It Matters

1. **Enables streaming inference**: Without attention sinks, any KV cache eviction strategy that removes initial tokens causes catastrophic perplexity degradation. StreamingLLM makes indefinitely long inference practical with fixed, bounded memory.
2. **Explains a fundamental attention behavior**: The attention sink phenomenon provides deep insight into how softmax-based attention actually operates in practice -- revealing that a significant portion of the attention mechanism serves a "housekeeping" role (distributing excess mass) rather than processing semantic content.
3. **Informs architecture design**: Understanding sinks has influenced the design of models with dedicated sink tokens at pre-training time, explicit attention-absorbing positions, and architectures that handle the excess-attention problem by design rather than by accident.
4. **Practical memory savings**: A fixed-size KV cache (e.g., 4 sink tokens + 4092 rolling window tokens) uses constant memory regardless of conversation length, critical for deploying multi-turn chat systems, streaming applications, and long-running agents.
5. **Connects to broader attention research**: The sink phenomenon is deeply related to Vision Transformer registers, differential attention, and other work on understanding and improving attention pattern quality across modalities.

## Key Technical Details

- **Perplexity without sinks**: When the first token's KV entry is evicted from a sliding window cache, perplexity jumps from roughly 10 to over 1000 across all tested models, regardless of window size. This is not a gradual degradation -- it is a catastrophic cliff.
- **Perplexity with sinks**: With 4 sink tokens + 4092 rolling window entries, perplexity stays within 0.1-0.3 of the full-context baseline, tested up to 4 million tokens on Llama 2 7B and other models.
- **Consistency across architectures**: The attention sink pattern and StreamingLLM fix work on Llama 2 (7B, 13B, 70B), Falcon (7B, 40B), Pythia (6.9B, 12B), and MPT (7B, 30B) -- every autoregressive transformer tested.
- **Attention mass concentration**: In typical attention heads, 20-80% of total attention mass may be directed at the first token for queries that lack highly relevant context in their window.
- **No retraining required**: StreamingLLM is a pure inference-time technique. No model weight modifications, no additional training, and no architectural changes are needed.
- **Position re-indexing requirement**: When using RoPE (Rotary Position Embeddings), the sink tokens must keep positions 0-3 and the rolling window tokens must be re-indexed to start at position 4 with no gaps. Positional gaps cause RoPE frequency mismatches that severely degrade generation quality.
- **Pre-training with sinks**: Some newer models are pre-trained with dedicated learnable sink tokens, improving attention sink efficiency and reducing the reliance on arbitrary first-token content serving as an accidental drain.

## Implications for Model Architecture and Training

Understanding attention sinks has several practical implications for model design:

- **Dedicated sink tokens at pre-training**: Some newer models are pre-trained with explicit learnable sink tokens prepended to every sequence, ensuring the attention drain is optimized from the start rather than emerging accidentally in the first content token.
- **Sink-aware KV cache compression**: Any scheme that compresses or evicts KV cache entries (for example, H2O -- Heavy-Hitter Oracle) must account for attention sinks. Eviction policies based on attention score rankings will naturally preserve sink tokens because they are the most-attended positions.
- **Attention-free architectures**: State-space models (Mamba, S4) and linear attention variants do not use softmax and therefore do not exhibit attention sinks. This removes one source of inefficiency but introduces other tradeoffs in modeling quality.
- **Multi-turn conversation context**: In chat applications where multiple turns are concatenated, the "first token" sink is the very first token of the *entire* conversation. System prompts placed at the beginning naturally occupy the sink position, which is fortuitous for deployment.

## Common Misconceptions

- **"The first token receives high attention because it contains important information."** The high attention is content-independent. Experiments replacing the first token with random noise show the model still directs significant attention to position 0. The phenomenon is purely positional and mathematical, not semantic.
- **"Attention sinks are a bug that should be fixed."** They are an emergent solution to a real constraint: softmax must allocate all probability mass somewhere, and the model needs a consistent "nowhere" option. Some newer architectures now design explicit sink mechanisms, treating this as a feature to be optimized rather than a bug to be eliminated.
- **"StreamingLLM provides the same quality as full-context attention."** StreamingLLM maintains stable *perplexity* (the model does not degrade), but it cannot recover information from tokens that have been evicted from the cache. It cannot perform tasks requiring attention to specific content from the distant past if that content is no longer in the rolling window.
- **"This only matters for very long sequences."** The sink phenomenon exists at all sequence lengths -- even 512-token sequences show disproportionate attention to the first token. StreamingLLM is specifically needed for long-context streaming deployment, but the underlying attention pattern is universal.

## Connections to Other Concepts

- **Sliding window attention**: StreamingLLM is built directly on top of the rolling buffer KV cache from sliding window attention, adding persistent sink tokens to prevent the catastrophic failure that occurs when initial tokens are evicted.
- **KV cache management**: Understanding attention sinks is essential for any KV cache eviction or compression strategy. Policies that remove initial tokens will fail catastrophically; policies that preserve them (even just 1-4 entries) succeed.
- **Softmax and attention normalization**: The sink phenomenon is a direct consequence of softmax's normalization constraint requiring attention weights to sum to 1. Alternative attention normalization schemes (like sigmoid attention or linear attention) handle this differently and may not exhibit sinks.
- **Vision Transformer registers**: Darcet et al. (2023) independently identified a parallel phenomenon in Vision Transformers, where certain patch tokens accumulate excess attention. Their solution -- adding explicit learned "register" tokens -- is architecturally analogous to designing dedicated attention sinks.
- **Differential Transformer**: Addresses the same root cause (attention noise from softmax's sum-to-one constraint) through a different mechanism: subtracting two attention maps to cancel common-mode noise rather than absorbing it into designated sink positions.

## Further Reading

1. **"Efficient Streaming Language Models with Attention Sinks" (Xiao et al., 2023, arXiv:2309.17453)** -- The paper that identified and named the attention sink phenomenon, proposed StreamingLLM, and demonstrated stable inference over millions of tokens with fixed memory.
2. **"Vision Transformers Need Registers" (Darcet et al., 2023, arXiv:2309.16588)** -- Identifies the parallel attention sink phenomenon in Vision Transformers and proposes explicit register tokens as an architectural solution, providing important cross-domain validation.
3. **"Mistral 7B" (Jiang et al., 2023, arXiv:2310.06825)** -- Introduces the rolling buffer KV cache that StreamingLLM builds upon, demonstrating the practical sliding window inference infrastructure.
