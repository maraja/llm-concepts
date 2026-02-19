# Medusa and Parallel Decoding

**One-Line Summary**: Medusa adds multiple lightweight prediction heads to a base LLM, enabling parallel token generation and tree-structured verification to achieve 2-3x speedups without a separate draft model.

**Prerequisites**: Autoregressive decoding, speculative decoding, transformer architecture, KV caching, rejection sampling

## What Is Medusa?

Imagine an octopus writing with multiple arms simultaneously, each arm drafting the next few words in parallel rather than one word at a time. That is essentially what Medusa does to autoregressive language model inference. Standard LLM decoding generates one token per forward pass, leaving most of the GPU's compute capacity underutilized during the memory-bound decode phase. Medusa breaks this bottleneck by predicting several future tokens at once.

Medusa augments a pre-trained transformer with K additional prediction heads -- small MLP networks that share the same hidden state produced by the base model's final layer. While the original language model head predicts the next token at position t+1, Medusa head k predicts the token at position t+k+1. All heads operate in parallel on the same hidden representation, so the additional compute cost per forward pass is minimal. The candidates from these heads are then organized into a tree structure and verified in a single forward pass using a specially crafted attention mask.

This approach sits in the broader family of parallel decoding methods alongside speculative decoding, but with a critical architectural difference: there is no separate draft model. Everything lives within a single model, sharing a single KV cache, which eliminates the engineering complexity of managing two models and avoids vocabulary mismatch issues entirely. This simplicity is what makes Medusa particularly attractive for production deployments where operational complexity is a real concern.

## How It Works

### Multi-Head Architecture

The fundamental bottleneck Medusa addresses is the memory-bound nature of autoregressive decoding. During the decode phase, each forward pass generates only one token but must load the entire model's weights from GPU HBM into the compute units. The arithmetic intensity (FLOPs per byte loaded) is extremely low, meaning the GPU spends most of its time waiting for memory transfers rather than computing. Medusa amortizes this memory bandwidth cost across multiple token predictions per forward pass.

Each Medusa head is a lightweight MLP, typically one or two hidden layers with a residual connection, projecting from the base model's hidden dimension to the vocabulary size. For a 7B-parameter base model, each head contains roughly 10-50M parameters depending on configuration -- less than 1% of the base model. With K=4 heads, the total parameter overhead is approximately 1.5-3% of the base model size.

The architecture of a single Medusa head looks like this:

```
Input: h_t (hidden state from final transformer layer, dim=4096 for 7B model)
  -> Linear(4096, 4096) + SiLU activation
  -> Residual connection: output + h_t
  -> Linear(4096, vocab_size)  # project to vocabulary logits
Output: logits over vocabulary for position t+k+1
```

At inference time, all K heads process the final hidden state h_t simultaneously:

```
head_0(h_t) -> logits for position t+1  (original LM head)
head_1(h_t) -> logits for position t+2  (Medusa head 1)
head_2(h_t) -> logits for position t+3  (Medusa head 2)
...
head_K(h_t) -> logits for position t+K+1 (Medusa head K)
```

Each head produces a distribution over the vocabulary. The top-s candidates from each head are selected (typically s=5), and these candidates are organized into a tree of possible continuations. The Cartesian product would yield s^K total paths, but tree pruning based on joint probability keeps this manageable.

### Tree-Structured Verification

Rather than checking each candidate sequence independently (which would require multiple forward passes), Medusa constructs a tree attention mask that allows roughly 60-64 candidate sequences to be verified in a single forward pass. The tree structure exploits the fact that many candidates share common prefixes.

The custom attention mask ensures that each position in the candidate tree can only attend to its ancestors, maintaining causal consistency. This is implemented as a sparse binary mask matrix where entry (i, j) = 1 only if position j is an ancestor of position i in the tree. After the verification forward pass, the base model's own predictions are compared against the Medusa candidates. The longest matching prefix along any branch of the tree is accepted, and decoding resumes from that point.

For example, if the tree has depth 4 and the base model agrees with Medusa's candidates for the first 3 positions but disagrees at position 4, then 3 tokens are accepted in a single step -- a 3x improvement over standard autoregressive decoding for that particular step.

### Medusa-1 vs Medusa-2

**Medusa-1** freezes the base model backbone and only trains the additional heads. This is fast to train (4-8 hours on a single A100 GPU) and preserves the base model's weights exactly. The tradeoff is that the heads must learn to predict future tokens from hidden states that were never optimized for multi-step prediction. Medusa-1 achieves roughly 2.2-2.3x wall-clock speedup but produces approximate outputs -- the generation distribution differs slightly from the original model.

**Medusa-2** jointly fine-tunes both the base model and the Medusa heads together, then uses a rejection sampling scheme at inference time to guarantee that the output distribution matches the original model exactly. This combined training yields better head accuracy and achieves 2.3-3.6x speedup depending on the task and sampling temperature. The rejection sampling mechanism is adapted from standard speculative decoding: candidate tokens are accepted with probability min(1, p_base / p_medusa), ensuring distributional equivalence.

### EAGLE: A Related Approach

EAGLE (Extrapolation Algorithm for Greater Language-model Efficiency) takes a different angle on the same problem. Instead of predicting tokens directly from the final hidden state, EAGLE performs autoregression at the feature level -- predicting the next hidden state rather than the next token. This hidden state is then decoded to tokens through the existing LM head. Because hidden states are more predictable than tokens (they live in a continuous space rather than a discrete vocabulary), EAGLE achieves higher acceptance rates: typically 2.5-4x speedup, outperforming Medusa on most benchmarks.

The key architectural difference is that EAGLE's draft head takes as input both the current hidden state and the token embedding of the predicted token, creating an autoregressive chain at the feature level. This allows EAGLE to capture inter-token dependencies that Medusa's independent heads miss. The EAGLE-2 variant further improves on this by adding a confidence-based dynamic draft length, generating longer draft sequences when the model is confident and shorter ones when uncertain.

### Practical Deployment Considerations

When deploying Medusa in production, several practical factors affect real-world speedup. Batch size is critical: Medusa provides the most benefit at batch size 1 (single-request latency optimization) and diminishing benefit at large batch sizes where the system becomes compute-bound rather than memory-bound. The tree topology can be tuned per deployment -- a deeper tree works better for predictable text (code, structured data), while a wider, shallower tree suits open-ended generation. Medusa heads must be trained on data that matches the deployment domain: heads trained on general text will underperform on specialized domains like medical or legal text.

## Why It Matters

1. **Inference speedup without quality loss**: Medusa-2 achieves 2.3-3.6x faster generation while provably preserving the base model's output distribution through rejection sampling.
2. **Single-model simplicity**: Unlike standard speculative decoding, there is no separate draft model to train, deploy, or synchronize -- just a few extra MLP heads on the existing model.
3. **Minimal resource overhead**: The additional heads add only 1.5-3% memory overhead, and the single shared KV cache avoids the duplication required by two-model speculative decoding.
4. **Drop-in compatibility**: Medusa can be applied to any autoregressive transformer model without modifying the base architecture, making it applicable across model families and sizes.
5. **Training efficiency**: Medusa-1 heads can be trained in 4-8 hours on a single A100, making it accessible for teams that want faster inference without major infrastructure investment.

## Key Technical Details

- Each Medusa head is a 1-2 layer MLP with residual connection, ~10-50M parameters per head
- Typical configuration uses K=4 heads with top-s=5 candidates per head
- Tree attention mask allows ~60-64 candidate sequences verified per forward pass
- Medusa-1: frozen backbone, ~2.2-2.3x speedup, approximate distribution
- Medusa-2: joint fine-tuning + rejection sampling, 2.3-3.6x speedup, exact distribution
- Memory overhead: ~1.5-3% of base model parameters
- Training data: typically the same data used for fine-tuning the base model or a representative subset
- Training cost: ~4-8 hours on a single A100 for Medusa-1 heads on a 7B model
- Acceptance rate per head decreases with distance: head 1 ~70-80%, head 4 ~30-40%
- Average tokens accepted per step: ~2.5-3.5 depending on text domain and temperature
- Speedup is higher for code generation (~3x) than creative writing (~2x) due to higher predictability
- EAGLE uses feature-level autoregression on hidden states rather than vocabulary-level prediction, achieving 2.5-4x speedup with better acceptance rates
- Both Medusa and EAGLE are compatible with quantized models (GPTQ, AWQ)
- Optimal tree topology varies by use case: wider trees for diverse text, deeper trees for predictable sequences
- Medusa heads are model-specific and must be retrained when the base model changes or is updated

## Common Misconceptions

- **"Medusa requires a separate draft model like speculative decoding."** Medusa is a single-model approach. The prediction heads are tiny extensions of the base model sharing the same hidden states and KV cache, not an independent model.
- **"More heads always means more speedup."** There are diminishing returns beyond K=4-5 heads because acceptance rates drop sharply for more distant token predictions, and the tree verification cost grows with the number of candidates.
- **"Medusa changes the model's outputs."** Medusa-2 with rejection sampling provably preserves the original model's output distribution. Medusa-1 does produce approximate outputs, but the deviation is typically small and often imperceptible in practice.
- **"Parallel decoding only helps for greedy decoding."** Medusa works with both greedy and sampling-based generation, though speedup ratios tend to be higher at lower temperatures where head predictions are more accurate.
- **"The tree verification step is expensive."** The tree verification is a single forward pass with a modified attention mask. The additional compute from processing ~64 candidate positions is modest compared to the savings from accepting multiple tokens per step.

## Connections to Other Concepts

- **Speculative Decoding**: Medusa is a self-speculative variant that eliminates the need for a separate draft model while using the same verify-then-accept paradigm.
- **KV Caching**: Medusa benefits from sharing a single KV cache across all heads, unlike two-model speculative decoding which must manage separate caches.
- **Knowledge Distillation**: Medusa-1 head training can be viewed as distilling the base model's multi-step prediction capability into lightweight heads.
- **EAGLE**: A closely related parallel decoding method that operates on hidden state features rather than token-level predictions, often achieving higher acceptance rates.
- **Batch Inference**: Medusa's tree verification is complementary to batched inference -- the candidate tree adds modest compute to already-batched forward passes.
- **Quantization**: Medusa heads can be applied on top of quantized base models, combining memory savings from quantization with latency savings from parallel decoding.
- **Lookahead Decoding**: Another parallel decoding method that uses Jacobi iteration to generate and verify multiple tokens, providing a training-free alternative to Medusa's learned heads.
- **Continuous Batching**: In serving systems with continuous batching, Medusa's per-request speedup translates into higher overall throughput by reducing the time each request occupies a batch slot.

## Diagrams and Visualizations

![Medusa architecture diagram showing multiple prediction heads attached to the base model's final hidden state for parallel token generation](https://raw.githubusercontent.com/FasterDecoding/Medusa/main/assets/medusa_logo.png)
*See detailed architecture and tree verification diagrams at: [Medusa GitHub Repository](https://github.com/FasterDecoding/Medusa)*

![Tree-structured verification diagram showing how candidate tokens from Medusa heads form a tree verified in a single forward pass](https://cai-tianle.github.io/medusa-site/assets/overview.png)
*See diagram at: [Medusa Project Page](https://sites.google.com/view/medusa-llm)*

*See EAGLE feature-level autoregression diagram at: [EAGLE GitHub Repository](https://github.com/SafeAILab/EAGLE)*

## Further Reading

- Cai, T., Li, Y., Geng, Z., Peng, H., Lee, J. D., Chen, D., & Dao, T. (2024). "Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads." arXiv:2401.10774.
- Li, Y., Cai, T., Zhang, Y., Chen, D., & Dao, T. (2024). "EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty." ICML 2024.
- Leviathan, Y., Kalman, M., & Matias, Y. (2023). "Fast Inference from Transformers via Speculative Decoding." ICML 2023.
- Chen, C., Borgeaud, S., Irving, G., Lespiau, J.-B., Sifre, L., & Jumper, J. (2023). "Accelerating Large Language Model Decoding with Speculative Sampling." arXiv:2302.01318.
- Stern, M., Shazeer, N., & Uszkoreit, J. (2018). "Blockwise Parallel Decoding for Deep Autoregressive Models." NeurIPS 2018.
