# Sparse Attention

**One-Line Summary**: Sparse attention mechanisms restrict each token to attending to only a subset of other tokens rather than the full sequence, reducing attention's O(n^2) cost to O(n log n) or O(n) -- enabling practical processing of very long sequences.

**Prerequisites**: Understanding of self-attention and its quadratic complexity, multi-head attention, the attention score matrix and softmax, awareness of the memory and compute bottlenecks in long-sequence processing, familiarity with Flash Attention as a hardware-level optimization.

## What Is Sparse Attention?

Imagine reading a 500-page book. Full attention would be like cross-referencing every sentence with every other sentence in the entire book -- 250,000 sentences times 250,000 sentences equals 62.5 billion comparisons. Obviously, humans do not do this. You focus primarily on nearby sentences (local context), occasionally reference key passages from earlier chapters (global context), and skip vast amounts of irrelevant text.

Sparse attention applies this same principle to Transformers. Instead of computing attention between every pair of tokens (dense attention), sparse attention restricts which pairs of tokens can attend to each other. Each token attends to a carefully chosen subset of other tokens, dramatically reducing computation while preserving the ability to model both local patterns and long-range dependencies.

The fundamental insight is that most attention weights in a trained model are near zero anyway. Dense attention computes attention scores for all token pairs but then softmax concentrates most of the probability mass on a small number of relevant tokens. Sparse attention avoids computing the scores that would be near-zero, saving both compute and memory.

## How It Works

### Sparsity Patterns

Different sparse attention schemes define different rules for which tokens can attend to which:

**1. Sliding Window (Local) Attention**

Each token attends only to tokens within a fixed window of size w centered on its position:

```
Token i attends to tokens in range [i - w/2, i + w/2]
```

This captures local context efficiently with O(n * w) complexity. Mistral popularized this approach, using a window of 4,096 tokens.

The key insight is that stacking multiple layers creates an expanding receptive field. If the window size is w and the model has L layers, the effective receptive field grows to L * w tokens. A 32-layer model with a window of 4,096 has a theoretical receptive field of 131,072 tokens -- information from distant tokens propagates through intermediate layers.

```
Layer 1:  Each token sees w tokens
Layer 2:  Each token indirectly sees 2w tokens (via layer 1 representations)
Layer L:  Each token indirectly sees L*w tokens
```

**2. Dilated (Strided) Attention**

Tokens attend to every k-th token across the full sequence, creating a strided pattern:

```
Token i attends to tokens: i, i+k, i+2k, i+3k, ...
```

This provides long-range coverage with O(n^2 / k) complexity. Different heads can use different stride values, with some covering local context (small k) and others covering long-range context (large k).

**3. Global + Local Hybrid**

Certain designated "global" tokens attend to (and are attended by) all other tokens, while non-global tokens use local attention. This creates a communication highway through the global tokens:

- **Longformer** uses a combination of sliding window attention for most tokens plus global attention for special tokens (like [CLS]) or task-specific tokens.
- **BigBird** combines sliding window + global tokens + random sparse connections, provably preserving the expressive power of full attention.

```
Global tokens: attend to everything and are attended by everything
Local tokens:  attend to window + global tokens only
```

**4. Block Sparse Attention**

The sequence is divided into blocks, and attention is computed between blocks according to a predefined pattern. This is hardware-friendly because it maps well to GPU block operations:

```
Block pattern example (4 blocks):
Block 1 attends to: [Block 1, Block 2, Block 4]
Block 2 attends to: [Block 1, Block 2, Block 3]
Block 3 attends to: [Block 2, Block 3, Block 4]
Block 4 attends to: [Block 1, Block 3, Block 4]
```

**5. Adaptive / Learned Sparsity**

Instead of using a fixed pattern, the model learns which tokens to attend to. Methods include:

- **Hash-based (Reformer)**: Tokens are hashed based on their content; similar tokens (likely to have high attention) are grouped and attend to each other.
- **Top-k selection**: For each query, only the top-k highest attention scores are kept; the rest are zeroed out.
- **Routing-based**: A learned routing network decides which tokens should attend to which, similar to the router in Mixture of Experts.

### Hybrid Dense + Sparse Architectures

The most effective modern approach is not pure sparse or pure dense attention, but a hybrid combining both:

- **Jamba (AI21)**: Alternates between dense attention layers and Mamba (state space model) layers, achieving strong long-context performance with reduced compute.
- **Jamba-1.5**: Extends the hybrid approach to 256K context with a mixture of dense attention, sparse attention, and SSM layers.
- **MiniMax-01**: Uses sparse attention for most layers with periodic dense attention layers to ensure full-sequence information flow.

The pattern is: use dense attention where it matters most (early layers, periodic "full-refresh" layers), and sparse attention everywhere else.

### Complexity Comparison

| Method | Time Complexity | Memory Complexity | Range |
|---|---|---|---|
| Dense (standard) | O(n^2) | O(n^2) | Full |
| Sliding window | O(n * w) | O(n * w) | Local |
| Dilated | O(n * n/k) | O(n * n/k) | Strided global |
| Longformer | O(n * w + n * g) | O(n * w + n * g) | Local + global |
| BigBird | O(n * (w + g + r)) | O(n * (w + g + r)) | Local + global + random |
| Reformer | O(n * log n) | O(n * log n) | Content-based |

Where w = window size, g = number of global tokens, r = number of random connections, k = stride.

## Why It Matters

Sparse attention addresses the fundamental scalability bottleneck of Transformers:

1. **Long-context processing**: Dense attention at 128K tokens requires computing 16.4 billion attention scores per layer. Sliding window attention with w=4,096 requires only 524 million -- a 31x reduction. This is the difference between feasible and infeasible.

2. **Inference efficiency**: For production serving, sparse attention reduces both latency (fewer computations) and memory (smaller attention matrices), enabling faster and cheaper inference.

3. **Training efficiency**: Training on long sequences with dense attention is prohibitively expensive. Sparse attention makes it practical to pre-train on long documents without proportionally increasing compute.

4. **Quality preservation**: Research in 2025 shows that carefully designed sparsity patterns can maintain or even improve quality compared to dense attention at long contexts, because dense attention can be distracted by irrelevant distant tokens.

## Key Technical Details

- **FlashAttention and sparsity**: FlashAttention's block-based computation naturally supports block-sparse patterns. The combination of FlashAttention (hardware efficiency) and sparse attention (algorithmic efficiency) provides multiplicative benefits.
- **Attention sinks**: The first few tokens in a sequence receive disproportionately high attention regardless of content ("attention sinks"). Sparse attention patterns should always include these initial tokens; StreamingLLM leverages this by keeping a small window of initial sink tokens plus recent tokens.
- **Layer-wise heterogeneity**: Different layers benefit from different sparsity patterns. Early layers tend to need more local attention (processing token-level features), while later layers need more long-range attention (reasoning across the full context). Heterogeneous architectures assign different patterns to different layers.
- **KV cache interaction**: Sparse attention reduces the effective KV cache needed per token (only cached tokens within the attention pattern need to be stored), but managing variable-length cache entries per layer adds implementation complexity.
- **Lost in the middle**: Dense attention models struggle to retrieve information from the middle of long contexts. Some sparse attention patterns (like the global + local hybrid) can actually mitigate this by providing dedicated global tokens that maintain access to all positions.

## Common Misconceptions

- **"Sparse attention means information is lost."** Information propagates through multiple layers. Even with local-only attention, a 32-layer model can propagate information across the entire sequence. The question is whether propagation is sufficient for the specific task, not whether it occurs.
- **"Sparse attention is always an approximation of dense attention."** Some sparse patterns (BigBird) are provably as expressive as dense attention. Others make explicit trade-offs. The framing should be "different attention patterns for different needs," not "cheap approximation of the real thing."
- **"You should use the same sparsity pattern everywhere."** Hybrid approaches that mix dense and sparse layers consistently outperform uniform sparsity. The optimal architecture uses dense attention where it has the most impact and sparse attention elsewhere.
- **"Sparse attention is only for very long sequences."** Even at moderate sequence lengths (4K-8K), sparse attention can improve inference throughput by reducing the attention computation, which is especially valuable in high-throughput serving scenarios.
- **"FlashAttention makes sparse attention unnecessary."** FlashAttention is a hardware-level optimization that makes attention faster but does not change its O(n^2) algorithmic complexity. Sparse attention reduces the algorithmic complexity. They are complementary, not alternatives.

## Connections to Other Concepts

- **Flash Attention**: Flash Attention optimizes the hardware implementation of attention; sparse attention optimizes the algorithm. Combining both provides maximum efficiency.
- **Grouped Query Attention (GQA)**: GQA reduces the memory per token in the KV cache; sparse attention reduces the number of tokens in the attention computation. They are orthogonal optimizations.
- **Context Window Extension**: Sparse attention is one of the primary techniques enabling very long context windows. It complements positional encoding extensions (YaRN, RoPE scaling).
- **State Space Models (SSMs)**: SSMs achieve O(n) sequence processing without attention at all. Hybrid architectures combining SSM layers and sparse attention layers represent the state of the art for long-context models.
- **KV Cache**: Sparse attention patterns directly affect which KV cache entries need to be stored and accessed, influencing inference memory requirements.
- **Mixture of Experts**: Both MoE and sparse attention embody the principle of conditional computation -- not all parameters or computations are needed for every input.

## Diagrams and Visualizations

*Recommended visual: Comparison of dense, strided, and fixed sparse attention patterns from the Sparse Transformer paper — see [Generating Long Sequences with Sparse Transformers (arXiv:1904.10509)](https://arxiv.org/abs/1904.10509)*

*Recommended visual: BigBird attention pattern combining random, window, and global attention — see [BigBird Paper (arXiv:2007.14062)](https://arxiv.org/abs/2007.14062)*

## Further Reading

- Child et al., "Generating Long Sequences with Sparse Transformers" (2019) -- the foundational sparse attention paper from OpenAI, introducing strided and fixed sparse attention patterns.
- Beltagy et al., "Longformer: The Long-Document Transformer" (2020) -- introduces the sliding window + global tokens pattern that became widely adopted.
- Zaheer et al., "Big Bird: Transformers for Longer Sequences" (2020) -- proves that sparse attention with random, local, and global components preserves the expressive power of full attention.
- Jiang et al., "Mistral 7B" (2023) -- demonstrates that sliding window attention can achieve strong performance in a production-quality model, popularizing the approach.
