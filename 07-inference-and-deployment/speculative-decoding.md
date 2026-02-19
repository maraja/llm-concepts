# Speculative Decoding

**One-Line Summary**: Speculative decoding uses a small, fast "draft" model to guess multiple tokens ahead, then verifies all guesses in a single forward pass of the large "target" model, achieving 2-3x faster generation while producing output that is *mathematically identical* to standard decoding.

**Prerequisites**: Autoregressive generation, KV cache, token probability distributions, rejection sampling, the distinction between compute-bound and memory-bandwidth-bound operations.

## What Is Speculative Decoding?

Standard autoregressive generation has an uncomfortable truth: generating each token requires a full forward pass through the model, but during the decode phase, most of the GPU's computational power sits idle. The bottleneck is memory bandwidth -- reading billions of parameters from GPU memory -- not arithmetic. The GPU can multiply matrices far faster than it can load them.

Speculative decoding exploits this gap. Think of it like a junior associate drafting a document and a senior partner reviewing it. The junior works fast and produces a rough draft of several paragraphs. The senior reads the whole draft at once and approves most of it, only redlining a few sections. This is far faster than the senior dictating every word one at a time, because reading and approving in bulk is cheap compared to composing from scratch.

The "junior" is a small draft model (e.g., a 1B parameter model). The "senior" is the large target model (e.g., a 70B parameter model). The draft model proposes K tokens (typically 3-8), and the target model verifies all of them in one parallel forward pass.

## How It Works

### Step-by-Step Process

1. **Draft phase**: The small model autoregressively generates K candidate tokens: x_1, x_2, ..., x_K. This is fast because the draft model is small.

2. **Verification phase**: The target model processes the original context plus all K draft tokens in a **single forward pass**. This produces the target model's probability distributions P_target(x_t | x_1, ..., x_{t-1}) for each position.

3. **Acceptance/rejection**: For each draft token, compare the draft model's probability q(x) with the target model's probability p(x) using a modified rejection sampling scheme:

   For each position t from 1 to K:
   - If `p(x_t) >= q(x_t)`: Accept the token (the target model agrees or is even more confident).
   - If `p(x_t) < q(x_t)`: Accept with probability `p(x_t) / q(x_t)`. If rejected, resample from an adjusted distribution:

   $$x_t \sim \text{norm}\left(\max\left(0,\; p(x) - q(x)\right)\right)$$

   This adjusted distribution ensures the final token comes from exactly the target model's distribution.

4. **Advance**: All tokens up to (and including) the first rejection are kept, plus one newly sampled token at the rejection point. If all K tokens are accepted, the target model also provides one bonus token from its final-position distribution, yielding K+1 tokens from a single verification pass.

5. **Repeat**: Update the KV caches for both models and return to step 1.

### Why Output Quality Is Identical

This is the most remarkable property. The acceptance/rejection scheme is carefully constructed so that the marginal distribution of each accepted token is exactly p_target. This is not an approximation -- it is a formal guarantee based on the mathematics of rejection sampling.

The key insight: when we reject a draft token, we do not simply pick the target model's top token. We sample from the *residual* distribution (p - q, normalized), which mathematically ensures the overall process produces samples from p_target. This holds for any draft model quality, though better draft models mean higher acceptance rates and thus greater speedup.

### The Speedup Calculation

Let alpha be the average acceptance rate per token. With a draft length of K, the expected number of tokens generated per verification step is:

$$E[\text{tokens per step}] = \frac{1 - \alpha^{K+1}}{1 - \alpha}$$

The speedup depends on:
- **Acceptance rate (alpha)**: How well the draft model approximates the target. Typical values: 0.6-0.85.
- **Draft length (K)**: More drafts means more potential tokens per step, but also more wasted compute on rejected tokens.
- **Speed ratio**: How much faster the draft model is compared to the target model.

For a typical setup (70B target, 1B draft, alpha = 0.7, K = 5), speedups of 2-3x are common.

## Why It Matters

Speculative decoding addresses a fundamental inefficiency in LLM serving. During the decode phase, large models are severely memory-bandwidth-bound -- GPUs are typically operating at only 1-5% of their peak FLOP capacity. Speculative decoding fills this computational gap without any quality compromise.

This is especially valuable for:
- **Interactive applications** where latency directly affects user experience.
- **Large model deployment** where the target model is expensive and slow per token.
- **Edge cases** where you cannot afford to use a smaller model due to quality requirements but need the speed.

The "free lunch" nature of speculative decoding -- identical quality, strictly faster -- makes it one of the most elegant optimizations in the LLM inference stack.

## Key Technical Details

- **Draft model selection**: The draft model should share the same vocabulary as the target model. Common pairings: Llama 3 70B with Llama 3 8B, or a purpose-trained small model.
- **Self-speculative decoding**: Some approaches use the target model itself with early exit (skipping later layers) as the draft model, avoiding the need for a separate model entirely.
- **Medusa and EAGLE**: These methods attach lightweight prediction heads to the target model that predict multiple future tokens simultaneously, eliminating the need for a separate draft model.
- **Tree-structured speculation**: Instead of a single draft sequence, generate a tree of possible continuations and verify all branches in one pass using careful attention masking. This increases the expected number of accepted tokens.
- **Batch interaction**: Speculative decoding is most beneficial at low batch sizes. At high batch sizes, the GPU is already well-utilized and the overhead of draft + verify becomes less worthwhile.
- **Optimal draft length**: K is a tunable hyperparameter. Too short means little benefit; too long means wasted draft compute on later tokens that are unlikely to be accepted (acceptance probability compounds).
- **KV cache management**: Both the draft and target models maintain separate KV caches. On rejection, the target model's KV cache must be rolled back to the rejection point.

## Common Misconceptions

- **"Speculative decoding uses an approximation."** No. The output distribution is *exactly* that of the target model. This is mathematically proven, not empirically approximated.
- **"A better draft model always helps."** A better draft model increases the acceptance rate, but if it is also slower, the net speedup may decrease. The draft model must be significantly faster than the target model for the scheme to pay off.
- **"Speculative decoding helps with all workloads."** It primarily helps during the decode phase (generating tokens one at a time). During the prefill phase (processing the prompt), the GPU is already compute-saturated and speculative decoding offers no benefit. It also helps less at large batch sizes.
- **"You need a separate trained draft model."** While this is the original approach, methods like Medusa, EAGLE, and self-speculative decoding avoid this requirement entirely.
- **"Rejected tokens are wasted."** Partially true, but the cost of draft tokens is low (small model), and even modest acceptance rates (60-70%) yield substantial speedups because verification of accepted tokens is essentially free.

## Connections to Other Concepts

- **KV Cache**: Speculative decoding requires careful KV cache management for both draft and target models. Roll-back on rejection is a key implementation detail.
- **Quantization**: A quantized target model is already faster per token, which reduces the relative benefit of speculative decoding. However, the two techniques can be combined.
- **Throughput vs. Latency**: Speculative decoding is primarily a latency optimization. At high batch sizes (throughput-oriented), its benefits diminish because the GPU is already well-utilized.
- **Model Serving Frameworks**: vLLM and TensorRT-LLM both support speculative decoding, with ongoing work to improve integration with continuous batching.
- **Knowledge Distillation**: A distilled small model can serve as an excellent draft model, combining two optimization strategies.

## Diagrams and Visualizations

*Recommended visual: Speculative decoding pipeline showing draft model generating candidate tokens and target model verifying in a single forward pass — see [Leviathan et al. Paper (arXiv:2211.17192)](https://arxiv.org/abs/2211.17192)*

*Recommended visual: Acceptance/rejection verification step showing how rejected tokens are resampled to maintain exact output distribution — see [Chen et al. Paper (arXiv:2302.01318)](https://arxiv.org/abs/2302.01318)*

## Further Reading

1. **"Fast Inference from Transformers via Speculative Decoding"** (Leviathan et al., 2023) -- One of the two foundational papers that independently proposed speculative decoding with its formal correctness guarantee.
2. **"Accelerating Large Language Model Decoding with Speculative Sampling"** (Chen et al., 2023) -- The other foundational paper, from DeepMind, with a clear presentation of the acceptance/rejection mathematics.
3. **"Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads"** (Cai et al., 2024) -- A practical alternative that adds parallel prediction heads to the target model itself, eliminating the need for a separate draft model.
