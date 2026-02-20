# Mixture of Experts (MoE)

**One-Line Summary**: Mixture of Experts is an architecture that replaces the dense feed-forward network with multiple parallel "expert" networks and a learned router that selects only a small subset of experts for each token, enabling models with vastly more parameters while keeping per-token computation constant.

**Prerequisites**: Understanding of feed-forward networks in Transformers, the concept of model parameters vs. compute (FLOPs), and why larger models generally perform better.

## What Is Mixture of Experts?

Imagine a hospital where every patient sees every specialist -- cardiologist, neurologist, dermatologist, orthopedist -- regardless of their condition. That would be absurdly expensive and wasteful. Instead, a triage nurse (the **router**) evaluates each patient and sends them to the relevant 1-2 specialists (the **experts**). The hospital can employ 100 specialists but each patient only consumes the time of 2.

![Mixture of Experts layer architecture showing a router/gating network that sends each token to a selected subset of expert feed-forward networks, with outputs weighted and combined](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/moe/00_switch_transformer.png)
*Source: [Mixture of Experts Explained -- Hugging Face Blog](https://huggingface.co/blog/moe)*


Mixture of Experts (MoE) applies this same logic to neural networks. Instead of one large feed-forward network (FFN) that processes every token, an MoE layer contains multiple FFNs (experts) and a **gating network** (router) that decides which experts each token should be sent to. Typically, only 1-2 experts are activated per token, so the computation cost remains similar to a dense model with a single FFN, while the total parameter count is multiplied by the number of experts.

This is the principle of **conditional computation**: different parts of the network are activated for different inputs. Not every parameter participates in every forward pass.

## How It Works


![Switch Transformer architecture showing top-1 routing where each token is dispatched to a single expert](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/moe/03_switch_layer.png)
*Source: [Mixture of Experts Explained -- Hugging Face Blog](https://huggingface.co/blog/moe)*

### The MoE Layer (Replacing the FFN)

In a standard Transformer block:
```
x -> Attention -> FFN -> output
```

In an MoE Transformer block:
```
x -> Attention -> MoE Layer (Router + N Experts) -> output
```

The MoE layer replaces the single FFN with $N$ expert FFNs and a router.

### Step 1: The Router (Gating Network)

Given a token representation $x \in \mathbb{R}^{d_{model}}$, the router computes a score for each expert:

$$g(x) = \text{softmax}(\text{TopK}(W_r \cdot x))$$

where $W_r \in \mathbb{R}^{N \times d_{model}}$ is the router's weight matrix and TopK selects the $k$ highest-scoring experts (typically $k = 1$ or $k = 2$), setting all other scores to $-\infty$ before softmax.

The router is extremely lightweight -- just a single linear layer followed by top-k selection and softmax. Its output is a sparse vector of weights over the $N$ experts, with only $k$ non-zero entries.

### Step 2: Expert Computation

Each selected expert $i$ processes the token independently:

$$e_i(x) = \text{FFN}_i(x) = W_{2,i} \cdot \sigma(W_{1,i} \cdot x)$$

Each expert is a standard FFN (or SwiGLU FFN in modern models) with its own independent parameters.

### Step 3: Weighted Combination

The MoE layer output is the weighted sum of the selected experts' outputs:

$$\text{MoE}(x) = \sum_{i \in \text{TopK}} g_i(x) \cdot e_i(x)$$

where $g_i(x)$ is the router weight for expert $i$. Only the top-k experts contribute; all others have zero weight.

### The Full Picture

For a model with $N = 8$ experts and $k = 2$ (top-2 routing):
- The router selects 2 of 8 experts for each token.
- The model has 8x the FFN parameters of a dense model.
- Each token uses only 2x the FFN computation of a dense model (actually 2x the FFN compute, but the FFN is typically ~2/3 of total compute, so the overall increase is moderate).
- Different tokens in the same sequence can be routed to different experts.

![Comparison of dense Transformer FFN vs. MoE layer showing how the single FFN is replaced by N parallel expert FFNs with a learned gating mechanism selecting top-k experts per token](https://upload.wikimedia.org/wikipedia/commons/thumb/4/4c/Sparse_MoE_with_Top-2_Gating.svg/800px-Sparse_MoE_with_Top-2_Gating.svg.png)
*Source: [Mixture of Experts -- Wikipedia](https://en.wikipedia.org/wiki/Mixture_of_experts)*


## Why It Matters

### More Parameters Without More Compute

The fundamental insight of MoE is decoupling **parameter count** from **computation cost**. Scaling laws show that larger models (more parameters) perform better. But more parameters normally means more computation per token. MoE breaks this link:

| Property | Dense 7B Model | MoE 8x7B Model (e.g., Mixtral) |
|----------|-----------------|----------------------------------|
| Total parameters | 7B | ~47B (8 expert FFNs + shared attention) |
| Active parameters per token | 7B | ~13B (attention + 2 of 8 experts) |
| FLOPs per token | Proportional to 7B | Proportional to ~13B |
| Quality | Good | Significantly better |

Mixtral 8x7B achieves performance comparable to dense models 2-3x its active parameter count. This is the "free lunch" of MoE: better quality at similar (not identical) inference cost.

### Inference Efficiency Challenges

MoE is not without trade-offs. While FLOPs per token are controlled, **memory** requirements are not:
- All $N$ expert FFNs must be stored in GPU memory, even though only $k$ are used per token.
- The total model size (and memory footprint) scales with the number of experts.
- This creates challenges for deployment, especially on consumer hardware.

### Load Balancing: The Central Challenge

A critical problem: the router might learn to send all tokens to just 1-2 "favorite" experts, leaving the rest unused. This **expert collapse** wastes parameters and defeats the purpose of having multiple experts.

Solutions include:

**Auxiliary load-balancing loss**: An additional loss term that penalizes uneven expert utilization:

$$\mathcal{L}_{\text{balance}} = \alpha \cdot N \sum_{i=1}^{N} f_i \cdot P_i$$

where $f_i$ is the fraction of tokens routed to expert $i$ and $P_i$ is the average router probability for expert $i$. This loss encourages the router to distribute tokens evenly.

**Expert capacity**: Set a maximum number of tokens each expert can process per batch. Tokens that exceed capacity are either dropped or sent to a secondary expert. The Switch Transformer uses this approach.

**Sinkhorn routing**: Use the Sinkhorn algorithm to find an optimal assignment that balances load while respecting token preferences.

## Key Technical Details

- **Expert count**: Typical values are 8 (Mixtral), 16, 64, or even 128 (Switch Transformer experimented with up to 2048). More experts means more total parameters but potentially harder load balancing.
- **Top-k**: Usually $k = 1$ (Switch Transformer) or $k = 2$ (Mixtral, GShard). Top-1 is more efficient; top-2 is more stable and higher quality.
- **Which layers are MoE**: Not all layers need to be MoE. Some architectures alternate dense and MoE layers (e.g., every other layer is MoE). Others make every layer MoE.
- **Shared components**: The attention layers are typically **shared** (not expertized) across all tokens. Only the FFN is replaced with the MoE layer. This is because attention computes interactions between tokens and benefits from being consistent.
- **Expert specialization**: Research shows that experts do specialize to some degree -- some may focus on certain languages, topics, or syntactic patterns -- but specialization is less clean-cut than the "each expert is a domain specialist" metaphor suggests.
- **Communication overhead**: In distributed training, MoE requires **all-to-all communication** to send tokens to their assigned experts across different GPUs. This is a significant engineering challenge at scale.

### Notable MoE Models

- **Switch Transformer** (Fedus et al., 2021): Demonstrated top-1 routing, scaling to over 1 trillion parameters. Showed that expert count can be scaled aggressively.
- **Mixtral 8x7B** (Mistral AI, 2023): The model that brought MoE to mainstream attention. 8 experts, top-2 routing, outperforming LLaMA 2 70B at lower inference cost.
- **Mixtral 8x22B** (Mistral AI, 2024): Scaled up version with larger experts.
- **DeepSeek-V2/V3** (DeepSeek, 2024): Innovated with fine-grained experts (up to 160 small experts) and novel routing strategies.
- **Grok** (xAI): Large-scale MoE model.

## Common Misconceptions

- **"MoE models are 8x faster than dense models of the same total parameter count."** MoE is faster than a dense model with the same *total* parameters, but not 8x faster. Active parameters are more than $1/N$ of total because attention layers are shared and fully dense. The speedup is typically 2-4x, not 8x.
- **"Each expert becomes a specialist in a clear domain."** Expert specialization exists but is fuzzy. An expert might process more tokens from certain languages or topics, but it also handles plenty of other tokens. The router's decisions are based on representation geometry, not clean semantic categories.
- **"MoE is always better than dense."** MoE models require more total memory, are harder to train (load balancing, communication), and can have higher inference latency per token (due to memory access patterns). Dense models are simpler, easier to deploy, and may be preferable when memory is constrained.
- **"Token dropping is fine."** When expert capacity is exceeded and tokens are dropped, those tokens receive degraded representations. This is a real quality concern, and modern architectures work hard to minimize or eliminate token dropping.
- **"The router has learned the optimal routing strategy."** Routing is still an active research area. Current routers are simple linear classifiers, and there is evidence that more sophisticated routing could improve quality significantly. Hash-based routing (no learned router) has also shown competitive results.

## Connections to Other Concepts

- **Feed-Forward Networks**: Each expert is a standard FFN; MoE replaces the single FFN with multiple experts (see `feed-forward-networks.md`).
- **Activation Functions**: Each expert uses the same activation function (typically SwiGLU) as a dense FFN would (see `activation-functions.md`).
- **Residual Connections**: The MoE layer's output is added to the residual stream, just like a dense FFN's output (see `residual-connections.md`).
- **Transformer Architecture**: MoE is a modification of the standard Transformer block, replacing one component (see `transformer-architecture.md`).
- **Next-Token Prediction**: MoE models are trained with the same next-token prediction objective as dense models (see `next-token-prediction.md`).
- **Logits and Softmax**: The router uses softmax to produce expert weights, similar in form to the output layer (see `logits-and-softmax.md`).

## Further Reading

- "Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer" -- Shazeer et al., 2017 (the foundational MoE paper for modern deep learning)
- "Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity" -- Fedus et al., 2021 (simplified routing, massive scale)
- "Mixtral of Experts" -- Jiang et al., Mistral AI, 2023 (the model that popularized MoE for open-weight LLMs)
