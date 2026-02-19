# KV Cache Compression

**One-Line Summary**: KV cache compression encompasses quantization, eviction, and token merging techniques that reduce the memory footprint of stored key-value states by 2-8x, making long-context inference (128K+ tokens) practically deployable on existing GPU hardware.

**Prerequisites**: KV cache, self-attention, grouped-query attention (GQA), quantization fundamentals, memory-bandwidth-bound operations, PagedAttention.

## What Is KV Cache Compression?

Imagine a historian who records every word of every conversation they have ever witnessed, in full detail, in massive leather-bound ledgers. Their library is bursting. KV cache compression is the set of strategies this historian can employ to reduce their storage: writing in shorthand instead of full script (quantization), discarding notes from forgettable conversations while keeping the important ones (eviction), or merging notes from similar conversations into consolidated summaries (token merging).

![H2O Heavy-Hitter Oracle diagram showing attention sink tokens, heavy hitters, and recent window retention strategy](https://raw.githubusercontent.com/FMInference/H2O/main/imgs/h2o_logo.png)
*See KV cache eviction strategy diagrams at: [H2O GitHub Repository](https://github.com/FMInference/H2O)*


The KV cache is the dominant memory consumer during LLM inference. For a 70B parameter model processing a 128K-token context, the KV cache alone can require over 40 GB of GPU memory -- more than the model weights themselves when quantized. This memory consumption scales linearly with both sequence length and batch size, creating a hard ceiling on how many long sequences a GPU can serve simultaneously.

Grouped-Query Attention (GQA) was the first major architectural change to address this, reducing the number of KV heads. But GQA is a training-time decision baked into the model architecture. KV cache compression techniques operate at inference time, stacking on top of GQA (or full MHA) to squeeze further reductions without retraining the model. The three main families -- quantization, eviction, and merging -- offer different trade-off profiles between memory savings, quality impact, and implementation complexity.

## How It Works


![StreamingLLM attention sink diagram showing how initial tokens act as attention sinks enabling infinite-length streaming](https://raw.githubusercontent.com/mit-han-lab/streaming-llm/main/figures/streaming_llm.png)
*Source: [StreamingLLM GitHub Repository (MIT-HAN-Lab)](https://github.com/mit-han-lab/streaming-llm)*

### KV Cache Quantization

Standard inference stores KV cache entries in FP16 (16 bits per value). Quantizing to lower precision reduces memory proportionally:

*See KIVI asymmetric quantization diagrams (per-channel keys vs per-token values) at: [KIVI Paper (arXiv:2402.02750)](https://arxiv.org/abs/2402.02750)*


| Precision | Bits per Value | Memory vs FP16 | Quality Impact |
|-----------|---------------|-----------------|----------------|
| FP16      | 16            | 1x (baseline)   | None           |
| INT8      | 8             | 2x reduction     | Negligible     |
| INT4      | 4             | 4x reduction     | Minor          |
| INT2      | 2             | 8x reduction     | Moderate       |

**KIVI** (Liu et al., 2024) achieves 2-bit KV cache quantization with minimal quality loss through a key insight: keys and values have different quantization-friendly axes.

- **Keys** are quantized **per-channel** (across the token dimension): each feature dimension has its own scale/zero-point, computed across all tokens. This works because key channels have relatively consistent magnitudes.
- **Values** are quantized **per-token** (across the channel dimension): each token position has its own scale/zero-point. This works because value vectors vary more across channels within a single token than across tokens within a single channel.

```python
# KIVI-style asymmetric quantization (simplified)
def quantize_keys_per_channel(keys, bits=2):
    # keys shape: [num_tokens, head_dim]
    # Compute scale per channel (along token dimension)
    k_min = keys.min(dim=0)  # [head_dim]
    k_max = keys.max(dim=0)  # [head_dim]
    scale = (k_max - k_min) / (2**bits - 1)
    keys_quantized = ((keys - k_min) / scale).round().to(int)
    return keys_quantized, scale, k_min

def quantize_values_per_token(values, bits=2):
    # values shape: [num_tokens, head_dim]
    # Compute scale per token (along channel dimension)
    v_min = values.min(dim=1)  # [num_tokens]
    v_max = values.max(dim=1)  # [num_tokens]
    scale = (v_max - v_min) / (2**bits - 1)
    values_quantized = ((values - v_min) / scale).round().to(int)
    return values_quantized, scale, v_min
```

INT8 KV cache quantization is widely supported in production frameworks (vLLM, TensorRT-LLM) and is nearly lossless for most models. INT4 is increasingly available with minor quality trade-offs. 2-bit approaches like KIVI are at the research frontier.

### KV Cache Eviction

Instead of keeping all tokens' KV entries, eviction strategies selectively discard low-importance entries. The challenge is determining which tokens are "important" for future attention.

**H2O (Heavy-Hitter Oracle)** (Zhang et al., 2023) identifies three categories of tokens that should be retained:

1. **Attention sink tokens**: The first few tokens (typically 1-4) in any sequence accumulate disproportionately high attention scores regardless of their content. These act as "sinks" that stabilize the attention distribution and must always be kept.
2. **Recent tokens**: The most recent window of tokens (local context) is almost always attended to and should be preserved.
3. **Heavy-hitter tokens**: Tokens that have historically received high cumulative attention scores across all previous generation steps. These are the semantically important tokens.

```
Full KV Cache (20 tokens):
[sink][t2][t3][t4][t5][t6][t7][t8][t9][t10][t11][t12][t13][t14][t15][t16][t17][t18][t19][t20]

H2O with 40% budget (8 tokens):
[sink]...[t7]......[t11]...[t14]...[t17][t18][t19][t20]
  ↑        ↑          ↑       ↑     ↑    ↑    ↑    ↑
  sink   heavy     heavy   heavy   recent window (4)
         hitter    hitter  hitter
```

H2O achieves strong performance with cache budgets as low as 20% of the full sequence length, meaning 80% of KV entries can be evicted with minimal quality degradation on many tasks.

**StreamingLLM** (Xiao et al., 2023) exploits the attention sink phenomenon for a different purpose: enabling infinite-length streaming inference. It keeps only the first few "sink" tokens plus a sliding window of recent tokens, allowing the model to process arbitrarily long streams without the KV cache growing unboundedly. Quality is good for local tasks but degrades for tasks requiring long-range retrieval.

### Token Merging

Rather than discarding KV entries entirely (losing information), token merging combines similar tokens' representations:

- Identify clusters of tokens with similar key vectors (using cosine similarity or learned routing).
- Merge each cluster's key and value vectors (via weighted averaging) into a single representative entry.
- The merged KV cache is smaller but retains more information than pure eviction.

This approach is less mature than quantization and eviction but represents a promising middle ground: the information is compressed rather than discarded.

## Why It Matters

1. **Enables long-context deployment**: A 70B model at 128K context requires ~40 GB for the KV cache at FP16. INT4 quantization reduces this to ~10 GB, making it feasible on a single 80 GB A100 alongside the model weights.
2. **Increases batch size**: Compressing the KV cache frees memory for more concurrent sequences, directly increasing throughput. A 4x KV cache reduction can enable roughly 4x more concurrent users.
3. **Essential for scaling**: As models move to 1M+ token contexts (Gemini 1.5 Pro, Claude 3.5), the KV cache at full precision would require hundreds of gigabytes. Compression is not optional at these scales -- it is mandatory.
4. **Complements other optimizations**: KV cache compression stacks with GQA, PagedAttention, prefix caching, and continuous batching. Each addresses a different dimension of the serving efficiency problem.
5. **Enables edge deployment**: On consumer GPUs (24 GB VRAM), aggressive KV cache compression is often the difference between a model fitting and not fitting for long conversations.

## Key Technical Details

- **Memory formula**: KV cache memory = `2 * num_layers * num_kv_heads * head_dim * seq_len * batch_size * bytes_per_value`. For Llama 3 70B (80 layers, 8 KV heads, 128 head_dim) at 128K context: `2 * 80 * 8 * 128 * 131072 * 2 bytes = ~42 GB` at FP16.
- **Asymmetric quantization is key**: Naive symmetric quantization of KV cache entries to low bit-widths causes significant quality loss. Asymmetric approaches (different quantization axes for K vs V) are essential for maintaining quality at INT4 and below.
- **Layer-wise importance varies**: Not all layers' KV caches are equally important. Some work applies higher precision to critical layers (often early and late layers) and more aggressive compression to middle layers.
- **Dynamic vs. static budgets**: H2O dynamically selects which tokens to keep at each generation step. Static approaches (fixed window + fixed sinks) are simpler but less adaptive.
- **Quality evaluation is task-dependent**: Eviction strategies perform well on summarization and general QA but can fail on needle-in-a-haystack retrieval tasks where the "needle" token might be evicted.
- **Framework support**: INT8 KV cache is supported in vLLM, TensorRT-LLM, and llama.cpp. INT4 support is emerging. More aggressive techniques (H2O, KIVI) currently require custom implementations.

## Common Misconceptions

- **"KV cache quantization is the same as weight quantization."** They are distinct operations. Weight quantization reduces model parameter storage; KV cache quantization reduces the runtime activation storage that grows with sequence length and batch size. A model can use INT4 weights with FP16 KV cache, or INT8 weights with INT4 KV cache, or any combination.
- **"Evicting tokens means the model forgets them completely."** The model has already processed those tokens during prefill -- their influence is baked into later tokens' representations through the residual stream. Eviction removes the ability to *directly attend* to those tokens in future steps, but their indirect influence persists.
- **"You should always use the most aggressive compression possible."** The right compression level depends on the task. Needle-in-a-haystack retrieval needs most of the KV cache; summarization tolerates aggressive eviction. Production systems should benchmark on representative workloads.
- **"GQA already solves the KV cache problem."** GQA reduces KV heads (e.g., from 64 to 8 in Llama 3 70B), giving roughly an 8x reduction compared to full MHA. But at 128K context lengths, even with GQA the KV cache is tens of gigabytes. Compression techniques stack on top of GQA for further reductions.

## Connections to Other Concepts

- **KV Cache**: This entire concept is about compressing the KV cache. A thorough understanding of what the KV cache stores, how it grows, and why it dominates inference memory is the essential prerequisite.
- **Grouped-Query Attention**: GQA is an architectural (training-time) approach to KV cache size reduction. Compression techniques are inference-time approaches that complement GQA.
- **PagedAttention**: Paged memory management handles memory *allocation* efficiency; compression handles memory *density*. They combine naturally -- compressed KV entries are stored in paged blocks.
- **Quantization (Weight)**: KV cache quantization shares mathematical foundations with weight quantization (scale factors, zero points, calibration) but operates on activations rather than parameters, requiring different strategies.
- **Flash Attention**: Flash Attention reduces the peak memory of attention *computation* (by tiling). KV cache compression reduces the persistent memory of attention *storage*. Both address attention memory costs but at different stages.

## Further Reading

- Liu et al., "KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache" (2024) -- The key paper on ultra-low-bit KV cache quantization with per-channel key and per-token value quantization.
- Zhang et al., "H2O: Heavy-Hitter Oracle for Efficient Generative Inference of Large Language Models" (2023) -- Introduces the heavy-hitter eviction strategy with attention sink preservation.
- Xiao et al., "Efficient Streaming Language Models with Attention Sinks" (2023) -- StreamingLLM, enabling infinite-length inference by exploiting attention sink tokens with a sliding window KV cache.
