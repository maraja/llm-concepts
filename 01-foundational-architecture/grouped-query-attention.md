# Grouped Query Attention (GQA)

**One-Line Summary**: Grouped Query Attention reduces the memory footprint of the key-value cache by sharing key-value heads across groups of query heads, achieving near-full-attention quality at a fraction of the memory cost -- making it the de facto standard for production LLM deployment.

**Prerequisites**: Understanding of multi-head attention (how queries, keys, and values are split across heads), awareness of the KV cache and why it matters for inference, basic understanding of inference memory bottlenecks, familiarity with the distinction between training cost and inference cost.

## What Is Grouped Query Attention?

Imagine a library with 32 researchers (query heads), each working on their own question. In the standard setup (multi-head attention / MHA), each researcher has their own personal copy of every book in the library (key-value heads). That is 32 complete copies of the entire library -- enormous space, but each researcher has exactly the reference they need.

In multi-query attention (MQA), the library cuts costs radically: there is only one copy of each book, and all 32 researchers share it. This saves massive space but creates a bottleneck -- the single set of reference materials must serve all 32 researchers, and quality suffers.

Grouped query attention (GQA) finds the middle ground: the library maintains 8 copies of each book, and groups of 4 researchers share each copy. This dramatically reduces space (8 copies instead of 32) while giving each researcher much better access than a single shared copy. The quality is nearly identical to having 32 copies, but at a quarter of the storage cost.

In LLM terms, GQA shares key-value (KV) heads among groups of query heads, reducing the size of the KV cache that must be stored in GPU memory during inference. This is not a quality optimization -- it is a memory optimization that makes it practical to serve large models with long contexts at scale.

## How It Works

### The Attention Variants

**Multi-Head Attention (MHA)**: The original Transformer attention. If there are H query heads, there are H key heads and H value heads. Each query head has its own dedicated key-value pair.

```
Queries:  Q_1, Q_2, Q_3, ..., Q_32    (32 heads)
Keys:     K_1, K_2, K_3, ..., K_32    (32 heads)
Values:   V_1, V_2, V_3, ..., V_32    (32 heads)
```

KV cache per token: 2 * H * d_head (where d_head is the per-head dimension).

**Multi-Query Attention (MQA)**: All query heads share a single key head and single value head.

```
Queries:  Q_1, Q_2, Q_3, ..., Q_32    (32 heads)
Keys:     K_shared                      (1 head)
Values:   V_shared                      (1 head)
```

KV cache per token: 2 * 1 * d_head. This is 32x smaller than MHA. But quality degrades because a single key-value representation must serve all 32 different query perspectives.

**Grouped Query Attention (GQA)**: Query heads are divided into G groups, with each group sharing one key-value head.

```
Group 1: Q_1, Q_2, Q_3, Q_4    share K_1, V_1
Group 2: Q_5, Q_6, Q_7, Q_8    share K_2, V_2
...
Group 8: Q_29, Q_30, Q_31, Q_32  share K_8, V_8
```

KV cache per token: 2 * G * d_head. With G=8 and H=32, this is 4x smaller than MHA but 8x larger than MQA.

### The KV Cache Problem

During autoregressive generation, every previously generated token's key and value vectors must be stored in memory (the KV cache) because each new token must attend to all previous tokens. The KV cache size is:

```
KV cache size = 2 * num_layers * num_kv_heads * d_head * sequence_length * batch_size * bytes_per_param
```

For a concrete example with a 70B-parameter model (Llama 2 70B):
- 80 layers, 64 query heads, 8 KV heads (GQA with G=8), d_head = 128
- At 16-bit precision with sequence length 4,096 and batch size 1:

```
MHA:  2 * 80 * 64 * 128 * 4096 * 2 bytes = ~10.7 GB
GQA:  2 * 80 *  8 * 128 * 4096 * 2 bytes = ~1.3 GB
```

GQA reduces the KV cache from 10.7 GB to 1.3 GB -- an 8x reduction. For longer sequences (128K tokens), the savings are proportionally larger, going from ~336 GB (impractical) to ~42 GB (feasible on a single high-end GPU).

### Converting MHA to GQA

A key practical result from Ainslie et al. (2023): existing MHA models can be converted to GQA by mean-pooling the key-value heads within each group and then fine-tuning for a small number of steps. This means GQA can be adopted without training from scratch:

```
K_group_g = mean(K_heads in group g)
V_group_g = mean(V_heads in group g)
```

Followed by 5-10% of original training compute for fine-tuning. Llama 2 70B was originally trained with MHA and converted to GQA this way.

### Choosing the Number of Groups

The number of KV groups G trades off between memory savings and quality:

| G (KV heads) | KV cache reduction | Quality relative to MHA |
|---|---|---|
| H (= MHA) | 1x | Baseline |
| H/4 (typical GQA) | 4x | ~99.5% |
| H/8 (aggressive GQA) | 8x | ~99% |
| 1 (= MQA) | Hx | ~97-98% |

The sweet spot for most models is G = H/4 to H/8. The quality loss is minimal, but the memory savings enable significantly higher throughput, longer contexts, or larger batch sizes.

## Why It Matters

GQA has become the standard attention configuration for production LLMs because it addresses the most critical bottleneck in real-world deployment:

1. **Inference throughput**: The KV cache is often the binding constraint on how many concurrent requests a server can handle. Reducing KV cache size directly increases serving throughput.

2. **Long context feasibility**: Processing 128K or 1M token contexts requires enormous KV caches. GQA makes long-context models practical to deploy without requiring excessive GPU memory.

3. **Cost efficiency**: GPU memory is the most expensive resource in LLM serving. GQA reduces memory requirements per request, translating directly to lower cost per token.

4. **Universal adoption**: Llama 2 70B, Llama 3 (all sizes), Mistral, Mixtral, Qwen, and most modern open-source and proprietary models use GQA. It is no longer an optimization choice -- it is the default.

## Key Technical Details

- **GQA is an inference optimization, not a training optimization**: During training, the KV cache is not used (since the full sequence is available). GQA reduces training compute slightly (fewer KV projection parameters) but the primary benefit is at inference time.
- **Batch size scaling**: The KV cache scales linearly with batch size. For high-throughput serving with large batches, the KV cache can dominate GPU memory. GQA's benefit is proportionally larger in high-batch-size scenarios.
- **Interaction with KV cache quantization**: GQA and KV cache quantization (storing KV cache in int8 or int4) are complementary optimizations. Combining GQA (fewer KV heads) with quantization (fewer bits per value) can reduce KV cache by 16-32x compared to full-precision MHA.
- **Interaction with Flash Attention**: FlashAttention supports GQA natively. The key-value heads are broadcast to match the number of query heads within each group during the tiled computation.
- **Per-layer variation**: Some architectures use different numbers of KV heads in different layers (more KV heads in early layers, fewer in later layers). This is an emerging optimization that matches KV capacity to each layer's needs.

## Common Misconceptions

- **"GQA degrades model quality significantly."** Extensive benchmarking shows that GQA with reasonable group sizes (4-8 query heads per KV head) produces quality indistinguishable from full MHA on standard benchmarks. The quality difference is typically within noise.
- **"GQA is the same as MQA."** MQA (all query heads share one KV head) is the extreme case of GQA (G=1). GQA with G > 1 consistently outperforms MQA because it provides more diverse key-value representations.
- **"GQA saves compute during training."** The savings are primarily in inference memory, not training compute. During training, the reduced parameter count provides minor compute savings, but this is not the motivation.
- **"GQA means the model has fewer parameters."** The total model parameter count is reduced slightly (fewer KV projection weights), but this is a negligible fraction of total parameters. A 70B model with GQA might have 69.5B parameters -- the difference is rounding error.
- **"GQA is only useful for large models."** While the absolute memory savings are largest for large models, GQA is beneficial at any scale where the KV cache is a bottleneck -- which includes smaller models deployed with long contexts or large batch sizes.

## Connections to Other Concepts

- **Multi-Head Attention**: GQA is a direct modification of MHA. Understanding MHA is essential for understanding what GQA changes and why.
- **KV Cache**: GQA's entire motivation is reducing KV cache size. Understanding the KV cache bottleneck is essential for understanding why GQA matters.
- **Flash Attention**: FlashAttention's memory-efficient attention computation and GQA's reduced KV heads are complementary optimizations that together enable long-context inference.
- **Quantization**: KV cache quantization and GQA are orthogonal optimizations that can be combined for maximum memory reduction.
- **Sparse Attention**: Both GQA and sparse attention reduce the effective cost of attention, but through different mechanisms: GQA reduces per-token memory; sparse attention reduces the number of tokens attended to.
- **Context Window Extension**: Longer contexts require larger KV caches, making GQA increasingly important as context lengths grow.
- **Model Serving / Throughput vs. Latency**: GQA is primarily a throughput optimization -- it enables serving more concurrent requests or longer contexts within the same memory budget.

## Further Reading

- Ainslie et al., "GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints" (2023) -- the original GQA paper, including the method for converting existing MHA models to GQA.
- Shazeer, "Fast Transformer Decoding: One Write-Head is All You Need" (2019) -- the original multi-query attention paper that motivated GQA.
- Touvron et al., "Llama 2: Open Foundation and Fine-Tuned Chat Models" (2023) -- describes the adoption of GQA in Llama 2 70B and its impact on inference efficiency.
