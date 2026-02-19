# Throughput vs. Latency Trade-offs

**One-Line Summary**: Throughput (how many total tokens the system produces per second) and latency (how quickly an individual user receives their response) are fundamentally competing objectives in LLM serving, and every deployment architecture involves conscious decisions about where to sit on this trade-off curve.

**Prerequisites**: Basic understanding of LLM inference (autoregressive generation), batching, GPU memory and compute basics, KV cache, the distinction between prefill and decode phases.

## What Is the Throughput-Latency Trade-off?

Imagine a highway. **Throughput** is the total number of cars that pass through per hour. **Latency** is how long it takes any single car to get from point A to point B. You can increase throughput by adding more cars (higher density), but at some point the highway becomes congested and every car slows down -- latency increases.

LLM serving faces exactly this tension. You can serve more users by packing more requests into each GPU batch, but each individual request takes longer because it shares GPU resources with others. Every production system must decide where on this spectrum to operate, and the right answer depends entirely on the application.

## How It Works

### The Three Key Metrics

**Time to First Token (TTFT)**: The time between receiving a user's request and delivering the first generated token. This is dominated by the prefill phase -- processing the entire input prompt through the model. TTFT is critical for interactive applications where users are watching a streaming response.

**Time Between Tokens (TBT)**: The interval between consecutive generated tokens during the decode phase. Also called inter-token latency. For a smooth streaming experience, TBT should be below ~50-80ms (faster than human reading speed). TBT increases when the GPU is shared across many concurrent requests.

**Tokens Per Second (TPS)**: This metric has two meanings depending on context:
- **Per-user TPS**: The rate at which a single user receives tokens. This is 1/TBT.
- **System TPS (Throughput)**: The total number of tokens generated across all users per second. This is the aggregate output of the entire system.

A system can have high aggregate TPS (processing many requests) while having low per-user TPS (each individual request is slow). This is the core of the trade-off.

### Why Batching Helps Throughput

During the decode phase, generating one token requires reading *all* model weights from GPU memory. This takes the same time whether you are generating for 1 request or 32 requests (up to a point), because the bottleneck is memory bandwidth, not compute.

With batch size 1:
$$\text{GPU utilization} = \frac{\text{Compute for 1 token}}{\text{Compute capacity}} \approx 1-5\%$$

The GPU is drastically underutilized. With batch size 32, the compute increases 32x but the memory read time is identical, so:

$$\text{System TPS} \approx 32 \times \text{Single-request TPS}$$

This near-linear scaling holds until the GPU becomes compute-bound or runs out of memory for KV caches.

### Why Batching Hurts Latency

The free lunch ends when resources become contended:

1. **Memory pressure**: Each request in the batch requires its own KV cache. More requests mean more memory consumed, leaving less room for other optimizations. Eventually, requests must wait for memory to free up.
2. **Compute contention**: At large batch sizes, the GPU transitions from memory-bandwidth-bound to compute-bound. The forward pass takes longer, increasing TBT for every request.
3. **Scheduling overhead**: With continuous batching, new requests entering the batch trigger prefill computations that momentarily increase TBT for existing decode-phase requests (the "prefill interference" problem).
4. **Queueing delay**: When the system is saturated, new requests wait in a queue, increasing TTFT dramatically.

The relationship is roughly:

$$\text{Latency} \propto \frac{\text{Batch Size}}{\text{GPU Bandwidth}} + \text{Queue Wait Time}$$

### Prefill-Decode Disaggregation

The prefill and decode phases have fundamentally different computational profiles:

| Characteristic | Prefill | Decode |
|---------------|---------|--------|
| Compute pattern | Compute-bound | Memory-bandwidth-bound |
| GPU utilization | High (60-90%) | Low (1-10%) |
| Parallelism | High (many tokens) | Low (one token per request) |
| KV cache | Writing (no cache yet) | Reading (growing cache) |

Mixing these phases on the same GPU creates interference: a long prefill blocks decode iterations for other requests, causing TBT spikes.

**Disaggregated serving** separates prefill and decode onto different GPU pools:
- **Prefill GPUs**: Optimized for high FLOPS, handle prompt processing, then transfer KV caches to decode GPUs.
- **Decode GPUs**: Optimized for memory bandwidth, handle token generation with consistent low TBT.

This eliminates prefill interference and allows each GPU pool to be independently scaled and optimized. The trade-off is added complexity and the overhead of KV cache transfer between GPUs.

### Application-Specific Priorities

**Chatbots and interactive assistants** (latency-sensitive):
- TTFT < 500ms (users notice delays above this).
- TBT < 50-80ms (for smooth streaming).
- Moderate batch sizes to keep latency low.
- Willing to sacrifice throughput for responsiveness.

**Batch processing and data extraction** (throughput-sensitive):
- Latency is largely irrelevant -- results are consumed later.
- Maximize batch size to saturate GPU utilization.
- Process thousands of documents with maximum aggregate TPS.
- Cost per token is the primary metric.

**Code completion** (extreme latency sensitivity):
- TTFT < 200ms (within the user's typing pause).
- Often only 1-5 tokens needed (single line completion).
- May use speculative decoding for speed.
- Willing to use more GPU resources per request.

**Retrieval-augmented generation (RAG)** (mixed):
- Long prefill (many retrieved documents in the prompt) means TTFT is naturally higher.
- TBT should still be fast for streaming the answer.
- Chunked prefill helps prevent TTFT from blocking other requests.

**Real-time voice agents** (ultra-low latency):
- End-to-end response time < 500ms for natural conversation.
- TTFT is the critical bottleneck.
- Batch size often 1 (dedicated GPU resources per conversation).
- Model size limited by latency requirement.

### Practical Architecture Decisions

**Single GPU, latency-optimized**:
- Small batch size (1-8).
- Quantized model to reduce per-token time.
- Speculative decoding for lower TBT.
- Good for: prototypes, low-traffic applications.

**Multi-GPU, balanced**:
- Tensor parallelism across 2-4 GPUs for lower latency per request.
- Moderate batch sizes (16-64).
- Continuous batching with vLLM or TGI.
- Good for: production chatbots, APIs with SLAs.

**Large cluster, throughput-optimized**:
- Many GPUs running independently (data parallelism).
- Large batch sizes (64-256+).
- Queue-based request management.
- Prefill-decode disaggregation.
- Good for: batch processing, high-traffic APIs.

**Routing architecture**:
- Small, fast model handles simple queries (low latency, low cost).
- Large, capable model handles complex queries (higher latency, higher quality).
- A classifier routes requests based on estimated difficulty.
- Good for: production systems optimizing cost and quality simultaneously.

## Why It Matters

The throughput-latency trade-off directly determines the **cost and quality of service** for LLM applications. Understanding it is essential for:

- **Capacity planning**: How many GPUs do you need to serve N concurrent users at P99 latency < X ms?
- **Cost optimization**: Throughput-optimized batch processing can be 10-50x cheaper per token than latency-optimized serving.
- **SLA design**: What latency guarantees can you make? What happens under load?
- **Architecture selection**: Should you use tensor parallelism (lower latency) or data parallelism (higher throughput)?

Getting this wrong means either overpaying for hardware (over-provisioned for latency) or delivering a poor user experience (under-provisioned, high latency under load).

## Key Technical Details

- **Tokens per second per dollar (TPS/$)** is arguably the most important metric for production systems. It captures both the hardware cost and the throughput achieved.
- **P50 vs. P99 latency**: Median latency (P50) is often excellent, but tail latency (P99) can be dramatically worse due to long-prompt prefill or garbage collection pauses. SLAs should specify P99.
- **Dynamic batching policies**: Serving frameworks can implement policies like maximum batch size, maximum wait time in queue, or priority queues for premium users.
- **Token budgets**: Some systems set per-request token budgets that limit maximum generation length, preventing long generations from consuming disproportionate resources.
- **Autoscaling**: Cloud deployments typically autoscale GPU instances based on queue depth or latency metrics, adding capacity during traffic spikes.

## Common Misconceptions

- **"Bigger batch size is always better for throughput."** Only up to the point where GPU memory is exhausted or compute becomes the bottleneck. Beyond that, throughput plateaus or even decreases due to OOM-induced request failures or swap overhead.
- **"Latency and throughput can both be optimized simultaneously."** Within a given hardware budget, improving one almost always degrades the other. You can shift the trade-off curve outward with better software (Flash Attention, continuous batching), but the trade-off itself never disappears.
- **"TTFT and TBT are equally important."** For most interactive applications, TTFT matters more for user perception of responsiveness. Users tolerate moderate TBT if the first token arrives quickly because streaming creates the illusion of fast response.
- **"Tensor parallelism always helps latency."** It reduces per-request compute time but adds communication overhead between GPUs. For small models or high-bandwidth interconnects this helps; for other scenarios the communication cost can negate the benefit.
- **"Throughput is just batch_size times per-request speed."** Real throughput is complicated by variable request lengths, KV cache memory fragmentation, prefill interference, and scheduling overhead. Actual throughput is typically 40-70% of the theoretical maximum.

## Connections to Other Concepts

- **KV Cache**: KV cache memory consumption is the primary constraint on batch size, which directly controls the throughput-latency trade-off. PagedAttention reduces waste and shifts the curve outward.
- **Flash Attention**: Speeds up both prefill and decode phases, benefiting both latency and throughput -- a rare win-win that shifts the entire trade-off curve.
- **Quantization**: Reduces model memory, freeing space for larger KV caches and thus larger batches. Also reduces per-token latency via lower memory bandwidth requirements.
- **Speculative Decoding**: A pure latency optimization that works best at low batch sizes, explicitly trading potential throughput for lower per-request latency.
- **Model Serving Frameworks**: The framework's scheduling policy, batching strategy, and memory management directly implement the chosen position on the throughput-latency trade-off curve.

## Diagrams and Visualizations

*Recommended visual: Throughput vs latency trade-off curve showing how increasing batch size improves throughput but degrades per-request latency — see [Lilian Weng – Large Transformer Model Inference Optimization](https://lilianweng.github.io/posts/2023-01-10-inference-optimization/)*

*Recommended visual: Roofline model showing compute-bound (prefill) vs memory-bound (decode) regimes — see [Efficient LLM Inference Survey (arXiv:2404.14294)](https://arxiv.org/abs/2404.14294)*

## Further Reading

1. **"Orca: A Distributed Serving System for Transformer-Based Generative Models"** (Yu et al., 2022) -- Introduced iteration-level scheduling (continuous batching), the foundational technique for navigating the throughput-latency trade-off in LLM serving.
2. **"Splitwise: Efficient Generative LLM Inference Using Phase Splitting"** (Patel et al., 2024) -- Explores prefill-decode disaggregation, demonstrating how separating the two phases onto different hardware improves both throughput and latency.
3. **"Efficiently Scaling Transformer Inference"** (Pope et al., 2022, Google) -- A thorough analysis of the compute and memory-bandwidth bottlenecks at different batch sizes and sequence lengths, providing a theoretical framework for understanding the trade-off.
