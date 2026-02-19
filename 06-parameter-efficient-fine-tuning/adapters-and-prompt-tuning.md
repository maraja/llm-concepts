# Adapters, Prefix Tuning & Prompt Tuning

**One-Line Summary**: Beyond LoRA, a family of parameter-efficient fine-tuning methods -- including bottleneck adapters, prefix tuning, prompt tuning, (IA)^3, and DoRA -- each offer distinct trade-offs in where and how they inject trainable parameters into a frozen pretrained model.

**Prerequisites**: Understanding of the transformer architecture (attention mechanism, feed-forward layers, residual connections), the basics of fine-tuning, familiarity with LoRA as a reference point for comparison, and the concept of key-value pairs in attention.

## What Is This Family of Methods?

Imagine a large orchestra performing a symphony. Full fine-tuning would be rewriting the entire score for every instrument. LoRA would be adding subtle harmony annotations to each musician's sheet music. But there are other approaches: you could insert a small chamber ensemble between movements (adapters), have a conductor signal new interpretive cues at the start of every passage (prefix tuning), whisper a brief instruction to the orchestra before they begin playing (prompt tuning), or adjust the volume knob on each instrument section (multiplicative rescaling).

Each of these PEFT methods represents a different philosophy about where to inject new learnable information into a frozen model. They share the goal of training only a tiny fraction of parameters while preserving most of the pretrained model's capability, but they differ significantly in mechanism, overhead, and quality.

## How It Works

### Bottleneck Adapters

Introduced by Houlsby et al. (2019), bottleneck adapters insert small trainable modules **between** existing transformer layers, specifically after the attention and feed-forward sub-layers, before the residual connection.

Each adapter module consists of:

```
Adapter(x) = x + f(x * W_down) * W_up
```

where:
- **W_down** projects from dimension d to a bottleneck dimension m (where m << d)
- **f** is a nonlinear activation (typically ReLU or GELU)
- **W_up** projects back from m to d
- The **residual connection** (x + ...) ensures the adapter can learn the identity function initially

With d = 4096 and m = 64, each adapter adds 2 x 4096 x 64 = 524,288 parameters. With two adapters per transformer layer (one after attention, one after FFN) and 32 layers, the total is roughly 33.5 million parameters for a model that might have 7 billion.

**Key characteristic**: Adapters add **sequential computation** to the forward pass. The input must pass through the adapter module, adding inference latency that cannot be eliminated by merging (unlike LoRA).

### Prefix Tuning

Introduced by Li and Liang (2021), prefix tuning prepends **learnable key-value vectors** to the attention mechanism at **every layer** of the transformer.

At each attention layer, instead of computing attention over just the input-derived keys and values, the model attends over:

```
K = [K_prefix; K_input]    (concatenation along sequence dimension)
V = [V_prefix; V_input]
```

where K_prefix and V_prefix are trainable parameter matrices of shape (p x d_head) per attention head, and p is the prefix length (typically 10-200 tokens).

The total trainable parameters are: p x num_layers x 2 x num_heads x d_head = p x num_layers x 2 x d_model.

In practice, the prefix parameters are not optimized directly. Instead, they are reparameterized through a small MLP during training (to stabilize optimization), and the MLP is discarded after training.

**Key characteristic**: Prefix tuning consumes part of the model's effective context window. A prefix of length 100 means 100 fewer tokens available for actual input. However, it adds minimal computational overhead beyond the slightly longer attention sequences.

### Prompt Tuning

Introduced by Lester et al. (2021), prompt tuning is the simplest PEFT method conceptually. It prepends a set of **learnable continuous embeddings** (soft prompts) to the input at the **embedding layer only** -- not at every layer like prefix tuning.

```
Input = [soft_prompt_1, soft_prompt_2, ..., soft_prompt_p, token_1, token_2, ..., token_n]
```

where each soft_prompt_i is a learnable vector of dimension d_model, and p is the prompt length.

Total trainable parameters: p x d_model. For p = 100 and d_model = 4096, that is just 409,600 parameters -- remarkably few.

**Key characteristic**: Prompt tuning is extremely parameter-efficient but tends to underperform other methods, especially on smaller models. However, the original paper showed a striking finding: as model scale increases (to 10B+ parameters), prompt tuning approaches the quality of full fine-tuning. This suggests that larger models are better at "interpreting" soft prompts.

### (IA)^3 -- Infused Adapter by Inhibiting and Amplifying Inner Activations

Introduced by Liu et al. (2022), (IA)^3 takes a minimalist approach: instead of adding new layers or parameters to the computation graph, it learns **element-wise rescaling vectors** that multiply existing activations.

Specifically, (IA)^3 learns three vectors:
- **l_k**: rescales the keys in attention (element-wise multiplication)
- **l_v**: rescales the values in attention
- **l_ff**: rescales the intermediate activations in the feed-forward network

```
K = l_k * (W_k * x)
V = l_v * (W_v * x)
FFN_intermediate = l_ff * (activation(W_up * x))
```

Total trainable parameters: num_layers x (d_model + d_model + d_ff), which is typically just a few hundred thousand parameters for the entire model.

**Key characteristic**: (IA)^3 has the fewest trainable parameters of any competitive PEFT method. It adds virtually zero inference overhead (element-wise multiplication is negligible). However, its expressiveness is limited, and it generally underperforms LoRA on complex tasks.

### DoRA -- Weight-Decomposed Low-Rank Adaptation

Introduced by Liu et al. (2024), DoRA decomposes the pretrained weight matrix into its **magnitude** and **direction** components, then applies LoRA only to the directional component:

```
W = m * (V / ||V||_c)
```

where:
- **m** is the magnitude vector (trainable, one scalar per output neuron)
- **V** is the directional matrix, where V = W_0 + B * A (pretrained weights plus LoRA adaptation)
- ||V||_c denotes the column-wise norm

During training, both the magnitude vector m and the LoRA matrices (B, A) are updated. This decomposition is inspired by weight normalization and aims to decouple "how much" (magnitude) from "which direction" (direction) the weight operates.

**Key characteristic**: DoRA consistently outperforms standard LoRA at the same rank, often matching full fine-tuning more closely. The additional overhead is minimal -- just one trainable scalar per output dimension per adapted layer.

## Why It Matters

This diversity of PEFT methods matters because **no single method dominates across all scenarios**:

- For production serving with adapter hot-swapping, LoRA's mergeability is unmatched.
- For extremely low parameter budgets on large models, prompt tuning or (IA)^3 may suffice.
- For maximum quality with PEFT constraints, DoRA offers improvements over standard LoRA.
- For research on how models process information, prefix tuning provides unique insights into attention steering.

The existence of multiple approaches also drives theoretical understanding of what makes fine-tuning work -- each method's success (or failure) reveals something about the geometry of pretrained weight spaces.

## Key Technical Details

| Method | Trainable Params (7B model) | Inference Overhead | Mergeable? | Quality (relative) |
|--------|---------------------------|-------------------|------------|-------------------|
| Bottleneck Adapters | ~30-60M | Moderate (sequential) | No | Good |
| Prefix Tuning | ~5-20M | Low (longer attention) | No | Good |
| Prompt Tuning | ~0.1-0.5M | Minimal | No (but trivially swappable) | Fair (improves with scale) |
| (IA)^3 | ~0.1-0.5M | Negligible | Yes (rescaling can be folded in) | Fair |
| LoRA | ~10-50M | Zero (when merged) | Yes | Very Good |
| DoRA | ~10-50M + magnitude | Zero (when merged) | Yes | Excellent |

## Common Misconceptions

- **"LoRA is strictly better than all alternatives."** LoRA has the best overall trade-off profile for most use cases, but adapters can outperform LoRA when inference overhead is acceptable, and DoRA outperforms LoRA at the same rank. The "best" method depends on constraints.
- **"Prompt tuning and prefix tuning are the same thing."** Prompt tuning modifies only the input embeddings. Prefix tuning modifies the key-value pairs at every attention layer. Prefix tuning is significantly more expressive and has more parameters.
- **"These methods are all obsolete because of LoRA."** Each method has active research and specific use cases. (IA)^3 is used in few-shot scenarios, prefix tuning in multi-task research, and DoRA is emerging as a LoRA successor.
- **"Adapter methods cannot be combined."** Methods like MAM Adapter (He et al., 2022) combine parallel adapters with prefix tuning, often outperforming either alone.
- **"Fewer parameters always means worse quality."** The relationship between parameter count and quality is nonlinear. A well-placed small adaptation (like (IA)^3's rescaling) can outperform a poorly configured adapter with more parameters.

## Connections to Other Concepts

- **LoRA and QLoRA**: LoRA is the most popular member of this PEFT family. Many innovations from other methods (like DoRA's magnitude decomposition) are designed as LoRA extensions.
- **Transfer learning**: All PEFT methods are forms of transfer learning, leveraging pretrained representations and adapting them efficiently.
- **Attention mechanism**: Prefix tuning directly manipulates the attention computation, offering insights into how attention heads can be steered.
- **Mixture of Experts (MoE)**: Some recent work treats multiple LoRA or adapter modules as "experts" that are routed to based on input, bridging PEFT with MoE architectures.
- **Continual learning**: PEFT methods are natural tools for continual learning, since the frozen base model prevents catastrophic forgetting of pre-existing knowledge.
- **Multi-task learning**: Methods like prompt tuning naturally support multi-task setups by learning different soft prompts for different tasks while sharing the same frozen backbone.

## Diagrams and Visualizations

*Recommended visual: Adapter bottleneck architecture showing small trainable modules inserted between frozen transformer layers — see [Houlsby et al. Adapter Paper (arXiv:1902.00751)](https://arxiv.org/abs/1902.00751)*

*Recommended visual: Comparison of adapter, prefix tuning, prompt tuning, and LoRA showing where each injects trainable parameters — see [Hugging Face PEFT Documentation](https://huggingface.co/docs/peft/index)*

## Further Reading

- **"Parameter-Efficient Transfer Learning for NLP"** -- Houlsby et al. (2019). The original bottleneck adapter paper that launched the PEFT research direction. [arXiv:1902.00751](https://arxiv.org/abs/1902.00751)
- **"Prefix-Tuning: Optimizing Continuous Prompts for Generation"** -- Li and Liang (2021). Introduces prefix tuning with learned key-value prefixes at every layer. [arXiv:2101.00190](https://arxiv.org/abs/2101.00190)
- **"DoRA: Weight-Decomposed Low-Rank Adaptation"** -- Liu et al. (2024). The most recent significant advance, decomposing weights into magnitude and direction for improved LoRA. [arXiv:2402.09353](https://arxiv.org/abs/2402.09353)
