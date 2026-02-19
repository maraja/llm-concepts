# ALiBi (Attention with Linear Biases)

**One-Line Summary**: ALiBi replaces learned positional embeddings with simple linear biases added directly to attention scores, enabling models to extrapolate to sequence lengths far beyond their training context with zero additional parameters and no fine-tuning.

**Prerequisites**: Understanding of self-attention mechanics (query-key dot products, softmax), positional encoding concepts (why Transformers need position information), familiarity with RoPE (Rotary Position Embeddings) as the dominant alternative, awareness of the context window extension problem.

## What Is ALiBi?

Imagine you are in a conversation with someone. Naturally, the words they just said are most relevant to understanding what they are saying now. Words from a minute ago are somewhat relevant. Words from an hour ago are barely relevant. You do not need an explicit memory of "this was word number 47 and this was word number 1,203" -- you just know that more recent things matter more, with a smooth decay.

ALiBi encodes exactly this intuition. Instead of giving the model an explicit representation of each token's position (like adding a "position 47" tag to the 47th token), ALiBi simply makes it harder for tokens to attend to distant tokens. It adds a penalty to the attention score that grows linearly with distance. Close tokens get a small penalty (easy to attend to). Distant tokens get a large penalty (harder to attend to). That is the entire mechanism.

The elegance of ALiBi is in what it does not do: it does not add positional embeddings to the input, it does not modify the token representations, and it does not require any learnable parameters. It is a fixed, deterministic bias applied to the attention computation. This simplicity is also why it can extrapolate -- there is no learned component that breaks when it encounters unseen positions.

## How It Works

### The Mechanism

In standard attention, the attention score between query token i and key token j is:

```
score(i, j) = q_i^T * k_j / sqrt(d_k)
```

ALiBi modifies this by subtracting a linear bias proportional to the distance between the two tokens:

```
score(i, j) = q_i^T * k_j / sqrt(d_k) - m * |i - j|
```

Where:
- `|i - j|` is the absolute distance between positions i and j
- `m` is a head-specific slope that determines how aggressively that attention head penalizes distance

The bias matrix for a sequence of length n looks like:

```
Head with slope m:

     j=0   j=1   j=2   j=3   j=4
i=0 [ 0    -m    -2m   -3m   -4m  ]
i=1 [-m     0    -m    -2m   -3m  ]
i=2 [-2m   -m     0    -m    -2m  ]
i=3 [-3m   -2m   -m     0    -m   ]
i=4 [-4m   -3m   -2m   -m     0   ]
```

After softmax, this linear penalty in log-space translates to an exponential decay in attention weight as a function of distance. Tokens nearby receive much more attention than distant tokens.

### Head-Specific Slopes

Different attention heads use different slopes, set deterministically (not learned):

```
For a model with n heads, slopes are geometric:
m_1 = 2^(-8/n), m_2 = 2^(-16/n), ..., m_n = 2^(-8)
```

For example, with 8 heads:
```
slopes = [1/2, 1/4, 1/8, 1/16, 1/32, 1/64, 1/128, 1/256]
```

Heads with small slopes (like 1/256) can attend across very long distances with minimal penalty -- these capture long-range dependencies. Heads with large slopes (like 1/2) focus sharply on nearby tokens -- these capture local patterns. The ensemble of heads provides both local precision and long-range reach.

### Why Extrapolation Works

The key property of ALiBi is that the bias function `m * |i - j|` is defined for any values of i and j. Unlike learned positional embeddings (which have no representation for positions beyond the training length) or RoPE (whose frequencies may produce unexpected patterns at unseen positions), ALiBi's linear penalty extends naturally to any sequence length.

At inference time, if the model was trained with sequences of length 1,024 and encounters a sequence of length 4,096, the bias values are simply larger (up to `m * 4096`) but the functional form is unchanged. The model has already learned how to use attention with linear distance penalties; longer sequences just extend the same pattern.

Empirical results show that ALiBi models trained on 1,024-token sequences can extrapolate to 2,048 or even 4,096 tokens with minimal quality degradation -- without any fine-tuning. This is a dramatic improvement over absolute positional embeddings, which completely fail outside their training range.

## Why It Matters

ALiBi matters for several reasons:

1. **Zero-shot length generalization**: The ability to process longer sequences at inference time without any additional training is practically valuable. It allows deployment flexibility -- a model trained at one context length can be used at longer lengths when needed.

2. **Simplicity and efficiency**: ALiBi adds no learnable parameters and requires no modification to the model architecture beyond adding a pre-computed bias matrix to the attention scores. Implementation is trivial compared to RoPE or other positional encoding schemes.

3. **Theoretical clarity**: ALiBi makes the inductive bias explicit: recent tokens are more relevant than distant ones, with a smooth, predictable decay. This is transparent and interpretable, unlike learned positional embeddings whose behavior is opaque.

4. **Influence on the field**: Although RoPE has become more dominant (largely due to the success of the Llama model family), ALiBi influenced models like MPT (MosaicML) and BLOOM (BigScience) and contributed to the broader understanding of positional encoding design.

## Key Technical Details

- **No position information in the residual stream**: Unlike RoPE (which rotates embeddings) or absolute positional embeddings (which are added to token embeddings), ALiBi does not modify the token representations at all. Position information exists only in the attention computation. This means the residual stream contains purely content-based representations.
- **Causal masking integration**: In autoregressive models, ALiBi is combined with the causal attention mask. Positions j > i (future tokens) are masked out entirely, and positions j <= i receive the distance-based penalty.
- **Computational cost**: The bias matrix can be pre-computed and cached. The only runtime cost is adding the bias to the attention logits, which is negligible compared to the QK^T matrix multiplication.
- **Relative position encoding**: ALiBi is a form of relative positional encoding (it depends on |i - j|, not on absolute positions i and j). This means the model's attention patterns are translation-invariant -- the same local pattern receives the same attention regardless of where it appears in the sequence.
- **Comparison with RoPE at long contexts**: While ALiBi naturally extrapolates, RoPE with extensions (YaRN, NTK-aware scaling) has been shown to achieve stronger performance at very long contexts (32K+ tokens) with fine-tuning. The trade-off is that RoPE requires extension techniques and fine-tuning, while ALiBi works out of the box.

## Common Misconceptions

- **"ALiBi means the model cannot attend to distant tokens."** The linear penalty makes distant attention harder but not impossible. Heads with small slopes can attend across the entire sequence with modest penalty. The model learns to route long-range information through these low-slope heads.
- **"ALiBi is strictly worse than RoPE."** For zero-shot extrapolation (no fine-tuning), ALiBi is superior to unmodified RoPE. RoPE with extensions outperforms ALiBi at very long contexts, but at the cost of additional complexity and fine-tuning.
- **"ALiBi captures no positional information."** The linear biases encode rich positional information -- specifically, relative distance between tokens. This is sufficient for most language tasks. The model knows which tokens are nearby and which are distant, even though it does not have explicit absolute position identifiers.
- **"ALiBi is obsolete."** While RoPE dominates current frontier models, ALiBi remains relevant for applications requiring robust extrapolation without fine-tuning, and its design principles (simple, deterministic, extrapolatable) continue to influence positional encoding research.

## Connections to Other Concepts

- **Rotary Position Embeddings (RoPE)**: The dominant alternative to ALiBi. RoPE encodes position by rotating embeddings in complex space; ALiBi adds linear biases to attention scores. They represent fundamentally different approaches to the same problem.
- **Context Window Extension**: ALiBi naturally supports extrapolation, reducing the need for context extension techniques. For RoPE-based models, extension methods (YaRN, position interpolation) are required.
- **Positional Encoding**: ALiBi is part of the broader positional encoding family. Understanding it requires and enriches understanding of sinusoidal, learned, and rotary alternatives.
- **Flash Attention**: FlashAttention implementations need to account for ALiBi biases in the tiled attention computation. Most FlashAttention libraries support ALiBi natively.
- **Sparse Attention**: ALiBi's distance penalty naturally creates a soft form of sparsity -- attention to very distant tokens is strongly suppressed. This connects to explicit sparse attention mechanisms that hard-cut attention beyond a window.

## Diagrams and Visualizations

![ALiBi attention bias matrix showing linear distance penalties added to attention scores, with different slopes per attention head](https://raw.githubusercontent.com/ofirpress/attention_with_linear_biases/master/alibi.png)
*Source: [Ofir Press – ALiBi GitHub Repository](https://github.com/ofirpress/attention_with_linear_biases)*

![Comparison of positional encoding methods: sinusoidal, rotary (RoPE), T5 bias, and ALiBi showing extrapolation performance on perplexity vs. sequence length](https://raw.githubusercontent.com/ofirpress/attention_with_linear_biases/master/alibi-extrapolation.png)
*Source: [Press et al., "Train Short, Test Long" – ALiBi Paper Repository](https://github.com/ofirpress/attention_with_linear_biases)*

*See also the ALiBi figure in the original paper: [Train Short, Test Long (arXiv:2108.12409)](https://arxiv.org/abs/2108.12409), Figure 1, which illustrates the head-specific slope mechanism and attention decay patterns.*

## Further Reading

- Press et al., "Train Short, Test Long: Attention with Linear Biases Enables Input Length Generalization" (2022) -- the original ALiBi paper, demonstrating zero-shot extrapolation and systematically comparing against other positional encoding methods.
- Workshop on Efficient NLP, "A Length-Extrapolatable Transformer" (2023) -- extends ALiBi ideas with additional techniques for improving extrapolation quality.
- Le Scao et al., "BLOOM: A 176B-Parameter Open-Access Multilingual Language Model" (2022) -- one of the largest models to use ALiBi, providing evidence for its effectiveness at scale.
