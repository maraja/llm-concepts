# Expert Parallelism

**One-Line Summary**: Expert parallelism distributes the experts of a Mixture-of-Experts (MoE) model across different GPUs, using all-to-all communication to route tokens to their assigned experts and back -- enabling models with trillions of total parameters (like Switch Transformer's 1.6T) while keeping per-token compute costs manageable through sparse activation.

**Prerequisites**: Mixture-of-Experts architecture (gating/routing mechanisms, sparse activation, top-k expert selection), data parallelism, tensor parallelism, pipeline parallelism, all-reduce and all-to-all communication primitives, and the concept of load balancing in distributed systems.

## What Is Expert Parallelism?

Mixture-of-Experts models activate only a small subset of their total parameters for each input token. Mixtral 8x7B has 46.7 billion total parameters but only uses 12.9 billion per token -- the router selects 2 of 8 experts for each token. This sparse activation means MoE models can be dramatically larger than dense models at the same computational cost per token.

But where do all those expert parameters physically reside? Each expert is a full feed-forward network (typically two large linear layers with an activation function), and with hundreds or thousands of experts, they cannot all fit on a single GPU. Expert parallelism solves this by distributing experts across devices: GPU 0 holds experts 0-7, GPU 1 holds experts 8-15, and so on. When a token is routed to expert 12, it must be physically sent to the GPU holding that expert, processed through the expert's feed-forward network, and sent back to its originating device.

Think of it like a hospital with specialist doctors spread across different buildings. A patient (token) arrives at triage (the router), which determines they need a cardiologist (expert 12) in Building B (GPU 1). The patient's records are transferred to Building B, the cardiologist treats them, and the results are sent back. The efficiency of the hospital depends critically on two things: making sure patients are distributed evenly across buildings so no single building is overwhelmed while others sit idle (load balancing), and making sure the transfer process is fast enough that doctors are not waiting for patients to arrive (communication overhead).

## How It Works

### Token Routing and All-to-All Communication

In expert parallelism, each device processes a local batch of tokens through the shared layers (attention, layer normalization, embeddings) using standard data or tensor parallelism. These shared layers are identical across all devices. At each MoE feed-forward layer, the following sequence occurs:

1. **Route**: The gating network (a small learned linear layer + softmax) computes routing decisions for all local tokens, determining which expert(s) each token should visit and with what weight.
2. **Dispatch (all-to-all)**: Tokens are sent to the devices holding their assigned experts. This is an all-to-all communication operation -- every device simultaneously sends tokens destined for other devices' experts and receives tokens that other devices' routers have assigned to its local experts.
3. **Compute**: Each device processes the received tokens through its local experts. Tokens assigned to different local experts are batched separately and processed in parallel.
4. **Combine (all-to-all)**: Processed tokens are sent back to their originating devices via a second all-to-all operation.
5. **Merge**: Each device combines the returned expert outputs, weighted by the gating scores computed in step 1, to produce the final MoE layer output for its local tokens.

The all-to-all communication pattern is what distinguishes expert parallelism from other parallelism strategies. Data parallelism uses all-reduce (aggregating identical gradients across replicas). Tensor parallelism uses all-reduce or all-gather (combining partial matrix products). Expert parallelism uses all-to-all (different tokens going to different places based on routing decisions), which is a fundamentally different and more irregular communication pattern.

### Load Balancing: The Central Challenge

If the router sends 80% of tokens to expert 3 and 2% to expert 7, the GPU holding expert 3 becomes a severe bottleneck while the GPU holding expert 7 sits mostly idle. Since the training step cannot complete until the most heavily loaded device finishes, load imbalance directly translates to wasted compute and wall-clock time. Several approaches address this:

**Auxiliary balance loss (standard approach)**: Add a differentiable load-balancing loss to the training objective:

$$\mathcal{L}_{\text{balance}} = \alpha \cdot N \cdot \sum_{i=1}^{N} f_i \cdot p_i$$

where $N$ is the number of experts, $f_i$ is the fraction of tokens actually routed to expert $i$ (computed from hard routing decisions), $p_i$ is the average routing probability assigned to expert $i$ by the gating network (computed from the soft gating scores), and $\alpha$ is a small coefficient (typically 0.01). Minimizing the product $f_i \cdot p_i$ encourages the gating network to spread tokens more evenly. The $N$ scaling ensures the loss magnitude is independent of the expert count.

**Capacity factors**: Each expert has a maximum processing capacity of $C \cdot (T/N)$ tokens per batch, where $T$ is the total number of tokens and $C$ is the capacity factor. Tokens routed to an already-full expert are either dropped (their output is replaced by the residual connection) or rerouted to a secondary expert. $C = 1.0$ allows no slack and may drop many tokens; $C = 1.25$ is a common compromise; $C = 1.5$ drops almost no tokens but wastes 50% of expert compute on padding empty slots.

**Expert Choice routing**: A fundamentally different paradigm -- instead of tokens choosing their top-$k$ experts, each expert selects its top-$k$ tokens from the full batch. This guarantees perfectly balanced load by construction, since every expert processes exactly the same number of tokens. The tradeoff is that some tokens may not be selected by any expert, and the routing is no longer a function of individual token features alone.

**Auxiliary-loss-free balancing (DeepSeek-V3)**: The auxiliary loss, while effective, contaminates the primary training gradient with a signal that has nothing to do with language modeling quality. DeepSeek-V3 introduces learnable bias terms added to the routing logits, with a separate update rule that adjusts these biases based on observed load imbalance. This decouples balance optimization from the language modeling gradient, avoiding the quality-balance tradeoff inherent in auxiliary losses.

### Practical Implementation Patterns

In production training systems, expert parallelism is implemented with careful attention to several engineering details:

- **Double buffering**: While one set of tokens is being processed by local experts, the next set is already being received via the network. This double-buffering hides communication latency behind computation.
- **Token padding and packing**: Tokens dispatched to each expert are padded to the capacity limit to enable efficient batched matrix multiplication on GPUs, which require regular tensor shapes. Excess capacity slots are filled with zero-padding.
- **Gradient accumulation**: Gradients from the expert computation are accumulated locally and synchronized with the broader data-parallel gradient all-reduce at the end of the training step.
- **Routing metadata**: The all-to-all communication requires exchanging not just token embeddings but also routing metadata (which tokens go where, with what weights), adding bookkeeping overhead.

## Why It Matters

1. **Enables trillion-parameter models**: Switch Transformer achieved 1.6 trillion parameters with 2,048 experts distributed across hundreds of devices. Each token activates only one expert, keeping per-token FLOPs comparable to a much smaller dense model.
2. **Favorable compute-memory tradeoff**: Expert parallelism allows models to be 5-10x larger in total parameters while keeping per-token FLOPs roughly constant, because only the routed experts are activated for each token.
3. **Communication can be overlapped**: The all-to-all communication pattern, while more complex than all-reduce, can be partially overlapped with non-MoE computation. With proper engineering, overhead is typically 10-20% of total training time.
4. **Foundation for frontier models**: Mixtral 8x7B, DeepSeek-V3 (256 experts + 1 shared), and other frontier models all rely on expert parallelism to scale efficiently beyond what dense architectures can achieve at the same compute budget.
5. **Adds a new scaling dimension**: Expert parallelism adds a fourth dimension to the standard 3D parallelism (data + tensor + pipeline), enabling "4D parallelism" strategies that distribute models across thousands of devices in the largest training runs.

## Key Technical Details

- **Mixtral 8x7B**: 8 experts per MoE layer, 2 active per token. 46.7B total parameters, 12.9B active per token. Matches or exceeds Llama 2 70B on most benchmarks at roughly 3x lower inference cost.
- **Switch Transformer**: 2,048 experts per MoE layer, 1 active per token. 1.6T total parameters. Demonstrated 4-7x pre-training speedup over dense T5 baselines of equivalent compute budget.
- **DeepSeek-V3**: 256 routed experts + 1 always-active shared expert per MoE layer, with 8 experts active per token. Uses the auxiliary-loss-free bias-based balancing approach to avoid quality degradation from balance losses.
- **Communication overhead**: All-to-all communication for expert parallelism typically adds 10-20% overhead to total training step time. This can be reduced through overlapping communication with non-MoE computation, or using hierarchical routing strategies.
- **Capacity factor tuning**: $C = 1.0$ is most compute-efficient but drops the most tokens. $C = 1.25$ is a common practical compromise. $C = 1.5$ drops almost no tokens but wastes significant compute on padding.
- **GShard scale**: 2,048 experts across 2,048 TPU v3 cores, with each core holding exactly one expert. Demonstrated scaling of a 600B MoE model for multilingual machine translation.

## Common Misconceptions

- **"Expert parallelism means each GPU runs exactly one expert."** In practice, each GPU typically holds multiple experts. With 256 experts on 32 GPUs, each GPU holds 8 experts. The 1:1 mapping (as in GShard's original design) is a special case optimized for very large expert counts.
- **"Load imbalance just wastes some compute."** Severe load imbalance causes synchronous GPU stalls -- the entire distributed training step waits for the most heavily loaded device to finish. In the worst case, one overloaded GPU can slow down the entire cluster to a fraction of its potential throughput.
- **"The auxiliary loss completely solves load balancing."** The auxiliary loss encourages but does not guarantee balance, and it introduces a quality-balance tradeoff -- pushing too hard for uniform distribution can force semantically unnatural routing decisions. This motivated DeepSeek-V3's auxiliary-loss-free approach.
- **"MoE models are always more efficient than dense models."** The communication overhead of all-to-all routing, the wasted compute from capacity factor padding, and the engineering complexity of load balancing mean that MoE models only become more efficient above a certain scale threshold where the sparse activation benefit outweighs these costs.

## Connections to Other Concepts

- **Mixture-of-Experts**: Expert parallelism is the distributed systems solution that makes large-scale MoE architectures practical. Understanding MoE routing, gating, and sparse activation is an essential prerequisite.
- **Data parallelism**: Often combined with expert parallelism -- the expert group is replicated across data-parallel groups, with each replica processing different micro-batches independently.
- **Tensor parallelism**: Can be applied within individual experts if they are large enough to benefit, and is standardly used for the shared (non-MoE) attention layers of the model.
- **Pipeline parallelism**: In deep MoE models, different layers can be distributed across pipeline stages, with expert parallelism applied within each stage that contains MoE layers.
- **3D/4D parallelism**: Expert parallelism adds a fourth scaling dimension orthogonal to the standard 3D parallelism (data + tensor + pipeline), enabling the comprehensive distribution strategies used in the largest training runs.

## Diagrams and Visualizations

![Switch Transformer architecture showing routing of tokens to individual experts across devices](https://production-media.paperswithcode.com/methods/Screen_Shot_2021-01-26_at_3.23.50_PM_dpkfwMF.png)
*Source: [Papers With Code - Switch Transformer](https://paperswithcode.com/method/switch-transformer)*

*Recommended visual: All-to-all communication pattern for expert parallelism showing token dispatch and combine operations across GPUs -- see [Lilian Weng - How to Train Really Large Models on Many GPUs](https://lilianweng.github.io/posts/2021-09-25-train-large/)*

*Recommended visual: MoE layer with gating network routing tokens to distributed experts, illustrating load balancing and capacity factors -- see [GShard paper (Lepikhin et al., 2020)](https://arxiv.org/abs/2006.16668), Figure 2*

## Further Reading

1. **"GShard: Scaling Giant Models with Conditional Computation and Automatic Sharding" (Lepikhin et al., 2020, arXiv:2006.16668)** -- Introduces expert parallelism at scale with automatic sharding of MoE models across thousands of TPU devices, establishing the foundational engineering patterns.
2. **"Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity" (Fedus et al., 2022, arXiv:2101.03961)** -- Simplifies MoE routing to top-1 expert selection and demonstrates expert parallelism at the 1.6T parameter scale with 2,048 experts.
3. **"Mixtral of Experts" (Jiang et al., 2024, arXiv:2401.04088)** -- Demonstrates the practical effectiveness of MoE with expert parallelism in a high-performing open-weight model, establishing strong quality benchmarks with favorable efficiency.
