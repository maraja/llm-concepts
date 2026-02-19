# Mixture of Depths

**One-Line Summary**: Mixture of Depths (MoD) dynamically routes each token at each layer through either the full transformer block or a skip connection, using a lightweight router to select only the top-k most important tokens for computation, reducing FLOPs by up to 50% while matching or exceeding standard transformer performance.

**Prerequisites**: Transformer architecture, residual connections, self-attention, feed-forward networks, mixture of experts (MoE), FLOPs and compute budgets, training dynamics.

## What Is Mixture of Depths?

Imagine a teacher grading a stack of essays. Some essays are clearly excellent or clearly failing -- a quick skim is enough. Others are borderline and need careful, line-by-line reading. A smart teacher spends the most time on the essays that need it, skimming the rest. A rigid teacher spends exactly 10 minutes on every essay regardless, wasting time on the easy ones.

Standard transformers are the rigid teacher. Every token passes through every layer, receiving the same amount of computation regardless of whether it is a semantically rich content word ("photosynthesis") or a trivial function word ("the"). Mixture of Depths introduces a per-token, per-layer decision: does this token need the full transformer block at this layer, or can it skip ahead through the residual connection with no computation?

The key result from Raposo et al. (Google DeepMind, 2024) is counterintuitive: MoD models trained with the same total FLOPs as standard transformers achieve *better* performance. The reason is that by saving computation on easy tokens at some layers, the FLOP budget can be redistributed to make the model larger (more layers, wider representations) while keeping the total compute constant. A bigger model that selectively skips work beats a smaller model that processes everything uniformly.

Mixture of Depths is orthogonal to Mixture of Experts (MoE), which varies the *width* of computation (which expert processes each token). MoD varies the *depth* (whether a token is processed at all at a given layer). The two can be combined for savings on both axes simultaneously.

## How It Works

### The Router Mechanism

At each MoD-enabled layer, a lightweight router (a single linear projection) takes each token's representation and produces a scalar score:

```python
# Router at layer l
# x shape: [batch_size, seq_len, d_model]
router_weights = nn.Linear(d_model, 1)  # Single scalar output
scores = router_weights(x).squeeze(-1)  # [batch_size, seq_len]
```

The top-k tokens (by router score) are selected to pass through the full transformer block. The remaining tokens skip the block entirely, passing through the residual connection unchanged:

```python
# Capacity ratio C = 0.5 means half the tokens are processed
k = int(C * seq_len)  # e.g., 512 out of 1024 tokens

# Select top-k tokens by router score
top_k_indices = scores.topk(k, dim=-1).indices  # [batch_size, k]

# Process only selected tokens through transformer block
selected_tokens = x[batch_indices, top_k_indices]  # [batch_size, k, d_model]
processed = transformer_block(selected_tokens)      # Full attention + FFN

# Scatter processed tokens back; unselected tokens keep their residual
output = x.clone()
output[batch_indices, top_k_indices] = processed + selected_tokens  # Residual
```

### Capacity Ratio and FLOP Savings

The **capacity ratio** C determines what fraction of tokens are processed at each MoD layer. With C = 0.5:

- Half the tokens skip the transformer block at each MoD layer.
- The self-attention and feed-forward computations operate on a sequence of length k = C * N instead of N.
- Since attention is quadratic in sequence length, attention FLOPs scale as (C * N)^2 = C^2 * N^2, giving a C^2 = 0.25x reduction for attention.
- Feed-forward FLOPs scale linearly: C * N, giving a 0.5x reduction.
- Overall FLOPs per MoD layer are approximately 50% of a standard layer (attention is a fraction of total layer compute, so the blended reduction is roughly C).

Not all layers need to be MoD-enabled. A typical configuration might apply MoD to every other layer, or to a subset of middle layers, keeping early and late layers as standard full-computation layers.

### Training with Auxiliary Losses

The router must learn which tokens benefit from computation at each layer. This is trained end-to-end with the standard language modeling loss, plus an auxiliary load-balancing loss to prevent the router from collapsing to always selecting the same tokens:

```
Total Loss = Language Modeling Loss + alpha * Load Balancing Loss
```

The load-balancing loss encourages the router to distribute selections across tokens rather than always choosing the same positional or token-type patterns. Without it, the router might learn trivial heuristics (e.g., always skip punctuation, always process the first token) that are suboptimal.

During training, the top-k selection is non-differentiable, so straight-through estimators or auxiliary prediction losses are used to provide gradients to the router weights.

### Which Tokens Get Skipped?

Analysis of trained MoD models reveals interesting routing patterns:

- **Function words** (articles, prepositions, conjunctions) are frequently skipped at middle layers -- their representations stabilize early.
- **Content words** in predictable contexts (e.g., "United States of **America**") are often skipped in later layers because their prediction is already confident.
- **Ambiguous or surprising tokens** consistently receive full computation across all layers.
- **Routing patterns vary by layer**: early layers tend to process most tokens (building initial representations), while middle and late layers are more selective.

## Why It Matters

1. **Same FLOPs, better performance**: The headline result is that MoD models trained with an identical FLOP budget as standard transformers achieve lower perplexity. By spending compute only where needed, the saved FLOPs can fund a larger model that performs better overall.
2. **Reduced inference cost**: At inference time, a MoD model with C = 0.5 uses roughly half the FLOPs per layer compared to a standard model of the same size. This translates directly to faster generation and lower serving costs.
3. **Orthogonal to MoE**: MoD (variable depth) and MoE (variable width) address different compute dimensions. Combining them creates models that are both wider and shallower where needed, compounding the efficiency gains.
4. **Adaptive computation**: MoD realizes a long-standing goal in deep learning: spending computation proportional to difficulty. Easy tokens in predictable contexts need less processing; complex, ambiguous tokens get the full treatment.
5. **Scaling implications**: As models grow to trillions of parameters, the ability to skip computation per-token becomes increasingly valuable. MoD suggests that future scaling laws should account for adaptive compute allocation, not just total FLOP counts.

## Key Technical Details

- **Capacity ratio tuning**: C = 0.5 is the commonly reported setting, but this is tunable. Lower C values (e.g., 0.25) give more aggressive FLOP savings but risk quality degradation. Higher values (0.75) give modest savings with minimal risk.
- **Router overhead**: The router itself is a single linear layer (d_model parameters per MoD layer). Its compute cost is negligible compared to the transformer block it gates.
- **Sequence ordering preservation**: Tokens that are processed and tokens that are skipped must be recombined in the correct sequence order. The scatter operation in the pseudocode above handles this, but efficient GPU implementations require careful indexing.
- **Attention over selected tokens**: When only k tokens are processed, attention is computed only among those k tokens. Unselected tokens are invisible to the attention mechanism at that layer. This is a form of dynamic sparse attention that emerges from the routing decision.
- **Comparison to early exit**: Early exit methods skip all remaining layers for a token. MoD is more flexible -- a token might skip layer 5 but be processed at layer 6. The routing decision is independent per layer.
- **Inference implementation**: At inference time, the router scores can be computed in advance for the entire sequence, and tokens can be partitioned into "process" and "skip" groups for efficient batched computation. However, for autoregressive generation, each new token's routing must be determined at each layer.
- **Training cost**: Training MoD models requires routing computation and auxiliary losses, adding approximately 5-10% overhead to training time compared to a standard transformer of the same size.

## Common Misconceptions

- **"Skipping layers means losing information."** Tokens that skip a layer still retain their full representation via the residual connection. The residual path passes the token's current state unchanged to the next layer. No information is lost -- the token simply does not receive additional processing at that layer.
- **"MoD is just dropout applied to layers."** Dropout randomly zeroes activations during training for regularization. MoD makes a learned, deterministic routing decision at inference time. The router is a trained component that identifies which tokens need computation, not a random mask.
- **"MoD and MoE are competing approaches."** They are complementary. MoE asks "which expert should process this token?" (varying width). MoD asks "should this token be processed at all at this layer?" (varying depth). A model can use both: route tokens to experts in layers where MoD selects them for processing, and skip them entirely in layers where MoD deems processing unnecessary.
- **"All tokens save the same amount of computation."** Different tokens may be selected at different layers. A function word might skip 6 out of 10 MoD layers while a complex content word skips only 1. The total compute per token varies, with the model naturally allocating more computation to harder tokens.

## Connections to Other Concepts

- **Residual Connections**: The skip pathway in MoD *is* the residual connection. Without residuals, skipping a layer would lose the token's representation entirely. Residual connections are what make MoD architecturally viable.
- **Mixture of Experts**: MoE varies computation width (which parameters are used); MoD varies computation depth (whether parameters are used at all). They compose naturally and address orthogonal efficiency dimensions.
- **Sparse Attention**: MoD can be viewed as a form of token-level sparse attention, where unselected tokens are dynamically excluded from the attention computation at each layer. This differs from pattern-based sparse attention (local windows, strided patterns).
- **Self-Attention**: At MoD layers, attention is computed only among the top-k selected tokens. This changes the effective receptive field dynamically, as different tokens "see" different subsets of the sequence at different layers.
- **Transformer Architecture**: MoD modifies the standard transformer by inserting a routing decision before each enabled layer, representing a significant architectural evolution from the fixed-computation-per-token paradigm.

## Diagrams and Visualizations

*Recommended visual: MoD architecture diagram showing the router selecting top-k tokens for full computation vs skip connection — see [Mixture of Depths Paper (arXiv:2404.02258)](https://arxiv.org/abs/2404.02258)*

*Recommended visual: Comparison of compute allocation across tokens showing how MoD dynamically skips easy tokens — see [Raphaël Millière's MoD Explainer](https://arxiv.org/abs/2404.02258)*

## Further Reading

- Raposo et al., "Mixture-of-Depths: Dynamically Allocating Compute in Transformer-Based Language Models" (Google DeepMind, 2024) -- The foundational paper demonstrating that MoD models match or exceed standard transformers at equivalent FLOP budgets.
- Fedus et al., "Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity" (2022) -- While focused on MoE, this paper establishes key concepts (router design, load balancing, capacity factors) that MoD builds upon.
- Graves, "Adaptive Computation Time for Recurrent Neural Networks" (2016) -- An early exploration of adaptive computation in neural networks, providing conceptual foundations that MoD realizes in the transformer setting.
