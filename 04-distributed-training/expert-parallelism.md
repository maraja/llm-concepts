# Expert Parallelism

**One-Line Summary**: Expert parallelism distributes the experts of a Mixture-of-Experts (MoE) model across different GPUs, using all-to-all communication to route tokens to their assigned experts and back -- enabling models with trillions of total parameters (like Switch Transformer's 1.6T) while keeping per-token compute costs manageable.

**Prerequisites**: Mixture-of-Experts architecture (gating/routing, sparse activation), data parallelism, tensor parallelism, pipeline parallelism, all-reduce and all-to-all communication primitives, and the concept of load balancing in distributed systems.

## What Is Expert Parallelism?

Mixture-of-Experts models activate only a small subset of their total parameters for each input token. Mixtral 8x7B has 46.7 billion total parameters but only uses 12.9 billion per token -- the router selects 2 of 8 experts for each token. This sparse activation means MoE models can be dramatically larger than dense models at the same computational cost.

But where do all those expert parameters physically reside? Each expert is a full feed-forward network, and with hundreds or thousands of experts, they cannot all fit on a single GPU. Expert parallelism solves this by distributing experts across devices: GPU 0 holds experts 0-7, GPU 1 holds experts 8-15, and so on. When a token is routed to expert 12, it must be physically sent to GPU 1, processed, and sent back.

Think of it like a hospital with specialist doctors spread across different buildings. A patient (token) arrives at triage (the router), which determines they need a cardiologist (expert 12) in Building B (GPU 1). The patient is transferred to Building B, treated, and returned. The efficiency of the hospital depends critically on two things: making sure patients are distributed evenly across buildings (load balancing) and making sure the transfer process is fast enough that doctors are not sitting idle (communication overhead).

## How It Works

### Token Routing and All-to-All Communication

In expert parallelism, each device processes a local batch of tokens through the shared layers (attention, normalization) using standard data or tensor parallelism. At each MoE layer, the following happens:

1. **Route**: The gating network computes routing decisions for all local tokens, determining which expert(s) each token should visit.
2. **Dispatch (all-to-all)**: Tokens are sent to the devices holding their assigned experts. This is an all-to-all communication operation -- every device sends tokens to every other device and receives tokens from every other device simultaneously.
3. **Compute**: Each device processes the received tokens through its local experts.
4. **Combine (all-to-all)**: Processed tokens are sent back to their originating devices. Another all-to-all operation returns the expert outputs.
5. **Merge**: Each device combines the expert outputs (weighted by gating scores) to produce the final MoE layer output for its local tokens.

The all-to-all communication pattern distinguishes expert parallelism from other parallelism strategies. Data parallelism uses all-reduce (same data, different samples). Tensor parallelism uses all-reduce or all-gather (different slices of the same computation). Expert parallelism uses all-to-all (different tokens going to different places).

### Load Balancing: The Central Challenge

If the router sends 80% of tokens to expert 3 and 2% to expert 7, the GPU holding expert 3 is a bottleneck while the GPU holding expert 7 sits idle. Load balancing is critical. Several approaches exist:

**Auxiliary loss (standard approach)**: Add a load-balancing loss to the training objective:

$$\mathcal{L}_{\text{balance}} = \alpha \cdot N \cdot \sum_{i=1}^{N} f_i \cdot p_i$$

where $N$ is the number of experts, $f_i$ is the fraction of tokens routed to expert $i$, $p_i$ is the average routing probability for expert $i$, and $\alpha$ is a small coefficient (typically 0.01). This loss penalizes uneven token distribution.

**Capacity factors**: Each expert has a maximum capacity of $C \cdot (T/N)$ tokens, where $T$ is the total tokens and $C$ is the capacity factor (typically 1.0-1.5). Tokens routed to an already-full expert are dropped or sent to a secondary expert. Higher capacity factors allow more imbalance but waste more compute on padding.

**Expert Choice routing**: Instead of tokens choosing experts, experts choose their top-$k$ tokens. This guarantees perfect load balance by construction -- every expert processes exactly the same number of tokens.

**Auxiliary-loss-free balancing (DeepSeek-V3)**: Rather than an auxiliary loss that can interfere with the primary training objective, DeepSeek-V3 uses learnable bias terms added to the routing scores. A separate update rule adjusts these biases to encourage balanced routing without contaminating the gradient signal for model quality.

### Scaling Expert Parallelism

Real-world MoE models combine expert parallelism with other parallelism strategies:

- **Expert parallelism + data parallelism**: Replicate the expert distribution across multiple groups of devices, each group processing a different data batch.
- **Expert parallelism + tensor parallelism**: Large individual experts can be split across devices using tensor parallelism within each expert.
- **Expert parallelism + pipeline parallelism**: Different layers of the model are on different pipeline stages, with expert parallelism applied within each MoE layer's stage.

## Why It Matters

1. **Enables trillion-parameter models**: Switch Transformer achieved 1.6 trillion parameters with 2,048 experts distributed across hundreds of devices, with each token activating only a small fraction.
2. **Favorable compute-memory tradeoff**: Expert parallelism allows models to be 5-10x larger in total parameters while keeping per-token FLOPs constant, because only the routed experts are activated.
3. **Communication efficiency**: The all-to-all communication pattern is well-suited to modern interconnects. With proper overlap of communication and computation, overhead can be kept to 10-20% of total training time.
4. **Foundation for frontier models**: Mixtral 8x7B, DeepSeek-V3 (256 experts + 1 shared), and Grok all rely on expert parallelism to scale efficiently.
5. **Complementary to other parallelism**: Expert parallelism adds a new dimension of scaling orthogonal to data, tensor, and pipeline parallelism, enabling the "4D parallelism" strategies used in the largest training runs.

## Key Technical Details

- **Mixtral 8x7B**: 8 experts, 2 active per token. 46.7B total parameters, 12.9B active per token. Matches or exceeds Llama 2 70B on most benchmarks at ~3x lower inference cost.
- **Switch Transformer**: 2,048 experts, 1 active per token. 1.6T total parameters. Demonstrated 4-7x pre-training speedup over dense T5 baselines.
- **DeepSeek-V3**: 256 routed experts + 1 shared expert, 8 active per token. Auxiliary-loss-free bias-based balancing to avoid quality degradation from the balancing loss.
- **Communication overhead**: All-to-all communication for expert parallelism typically adds 10-20% overhead. Can be reduced by overlapping with computation or using hierarchical routing (route to device groups first, then to individual experts).
- **Capacity factor tuning**: $C = 1.0$ (no extra capacity) is most efficient but drops the most tokens. $C = 1.25$ is a common compromise. $C = 1.5$ drops almost no tokens but wastes 50% of expert compute on padding.
- **GShard configuration**: 2,048 experts across 2,048 TPU v3 cores, with each core holding exactly one expert. Demonstrated scaling of a 600B MoE model for machine translation.

## Common Misconceptions

- **"Expert parallelism means each GPU runs one expert."** In practice, each GPU typically holds multiple experts. With 256 experts on 32 GPUs, each GPU holds 8 experts. The 1:1 mapping (GShard's design) is a special case.
- **"Load imbalance just wastes some compute."** Severe load imbalance causes GPU stalls -- the entire training step waits for the most heavily loaded device. In the worst case, one overloaded GPU can slow down the entire cluster.
- **"The auxiliary loss solves load balancing completely."** The auxiliary loss encourages but does not guarantee balance. It also trades off against model quality -- pushing too hard for balance can force semantically unnatural routing decisions. This motivated DeepSeek-V3's auxiliary-loss-free approach.
- **"MoE models are always more efficient than dense models."** The communication overhead of expert parallelism, the wasted compute from capacity factors, and the complexity of load balancing mean that MoE models only become more efficient above a certain scale threshold.

## Connections to Other Concepts

- **Mixture-of-Experts**: Expert parallelism is the distributed systems solution that makes large-scale MoE models practical. Understanding the MoE architecture (routing, gating, sparse activation) is essential.
- **Data parallelism**: Often combined with expert parallelism -- the expert group is replicated across data-parallel groups, with each replica processing different data batches.
- **Tensor parallelism**: Can be applied within individual experts if they are large, and is used for the shared (non-MoE) layers of the model.
- **Pipeline parallelism**: In deep MoE models, different layers can be distributed across pipeline stages, with expert parallelism applied within each MoE layer.
- **3D/4D parallelism**: Expert parallelism adds a fourth dimension to the standard 3D parallelism (data + tensor + pipeline), enabling the scaling strategies used in the largest models.

## Further Reading

1. **"GShard: Scaling Giant Models with Conditional Computation and Automatic Sharding" (Lepikhin et al., 2020, arXiv:2006.16668)** -- Introduces expert parallelism at scale with automatic sharding of MoE models across thousands of devices.
2. **"Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity" (Fedus et al., 2022, arXiv:2101.03961)** -- Simplifies MoE routing to top-1 selection and demonstrates expert parallelism at 1.6T parameter scale.
3. **"Mixtral of Experts" (Jiang et al., 2024, arXiv:2401.04088)** -- Demonstrates the practical effectiveness of MoE with expert parallelism in an open-weight model, establishing strong benchmarks with favorable efficiency.
