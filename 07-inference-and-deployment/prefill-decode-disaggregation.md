# Prefill-Decode Disaggregation

**One-Line Summary**: Prefill-decode disaggregation separates the compute-bound prefill phase (processing input tokens in parallel) and the memory-bandwidth-bound decode phase (generating tokens one at a time) onto different, independently optimized hardware pools, improving cost-efficiency by 1.5-2x and eliminating cross-phase interference.

**Prerequisites**: KV cache, prefill vs. decode phases, compute-bound vs. memory-bandwidth-bound operations, throughput vs. latency trade-offs, continuous batching, GPU architecture basics (FLOPS vs. memory bandwidth).

## What Is Prefill-Decode Disaggregation?

Imagine a restaurant with two fundamentally different types of work: prep cooking (chopping vegetables, making sauces, marinating proteins -- labor-intensive, parallelizable) and plating (carefully placing one element at a time on the dish -- precision-oriented, sequential). In a traditional kitchen, every chef does both tasks, switching between them. But prep cooking uses heavy equipment and physical strength, while plating needs steady hands and artistic precision. When a chef interrupts plating to do a burst of prep work, the plates in progress go cold. When a chef designed for heavy prep sits idle doing delicate plating, their strength is wasted.

*Recommended visual: Disaggregated prefill and decode phases running on separate GPU pools with KV cache transfer — see [Splitwise Paper (arXiv:2311.18677)](https://arxiv.org/abs/2311.18677)*


A smart restaurant separates these into two stations: a prep kitchen with powerful equipment and many hands, and a plating line optimized for precision and speed. Raw prepped ingredients are transferred between them. This is prefill-decode disaggregation.

In LLM inference, the prefill phase processes all input tokens in parallel through the model. This is a compute-bound operation -- the GPU's arithmetic units are the bottleneck, operating at high FLOP utilization. The decode phase generates output tokens one at a time, reading the entire model's weights from memory for each token. This is memory-bandwidth-bound -- the GPU's memory bus is the bottleneck, while its compute units sit mostly idle (often 1-5% utilization). These two phases have fundamentally different hardware requirements, yet in traditional serving they share the same GPUs and interfere with each other.

## How It Works


*Recommended visual: Interference between compute-bound prefill and memory-bound decode when colocated on the same GPU — see [DistServe Paper (arXiv:2401.09670)](https://arxiv.org/abs/2401.09670)*

### The Interference Problem

When prefill and decode operations share a GPU, several problems arise:

1. **Prefill latency spikes for decode**: A large prefill (e.g., 8K input tokens) takes significant GPU time. Any decode operations scheduled on the same GPU must wait, causing latency spikes in time-between-tokens (TBT) for in-flight generation requests. Users experience sudden pauses mid-response.

2. **Decode underutilizes compute**: During decode, each token generation reads the full model weights but performs relatively few FLOPs per byte loaded. The GPU's massive compute capacity (e.g., 312 TFLOPS on an H100) sits nearly idle. If the GPU were dedicated to decode, there is no way to reclaim this wasted compute.

3. **Conflicting optimization targets**: Prefill benefits from high compute throughput (larger matrix multiplications, higher batch sizes). Decode benefits from high memory bandwidth and low latency.

```
Shared GPU Timeline (interference):
Time: ──────────────────────────────────────────────→
GPU:  [prefill A][decode B,C,D][PREFILL E (large)][decode B,C,D stalled...][decode B,C,D]
                                    ↑
                          Decode latency spike: B,C,D users
                          experience multi-second pause

Disaggregated Timeline:
Prefill GPU: [prefill A][prefill E][prefill F][prefill G]...
Decode GPU:  [decode B,C,D][decode B,C,D][decode B,C,D][decode B,C,D]...
                 ↑ Smooth, uninterrupted token generation
```

### Architecture

Disaggregation separates the serving cluster into two pools:

**Prefill pool**: Optimized for compute throughput.
- Receives incoming requests, processes the full input prompt.
- Computes the KV cache for all input tokens.
- Transfers the computed KV cache to a decode instance.
- Can use GPUs with high FLOPS (e.g., H100 SXM) and large batch sizes for maximum compute utilization.

**Decode pool**: Optimized for memory bandwidth and latency.
- Receives the KV cache from prefill instances.
- Generates output tokens one at a time, appending to the KV cache.
- Returns completed responses to the client.
- Can use GPUs with high memory bandwidth, or even fewer/cheaper GPUs since compute requirements per token are low.

```
                    ┌─────────────┐
   Request ──────→  │ Prefill GPU │ ──── KV Cache Transfer ────→ ┌────────────┐
                    │ (compute    │     (NVLink / network)       │ Decode GPU │ ──→ Response
                    │  optimized) │                              │ (bandwidth │
                    └─────────────┘                              │  optimized)│
                                                                 └────────────┘
```

### KV Cache Transfer

The key engineering challenge is transferring the KV cache between pools efficiently. For a 70B model with a 4K-token prompt, the KV cache is roughly 1.3 GB (at FP16 with GQA). Transfer options include:

- **NVLink** (within a node): 900 GB/s on NVLink 4.0. The 1.3 GB transfer takes ~1.4 ms -- negligible compared to prefill time.
- **InfiniBand / RoCE** (across nodes): 400 Gb/s (50 GB/s). The 1.3 GB transfer takes ~26 ms -- noticeable but acceptable for many workloads.
- **Compression**: Quantizing the KV cache to INT8 before transfer halves the transfer time. Further compression is possible with INT4.

The transfer latency adds directly to the time-to-first-token (TTFT), so the distance between prefill and decode pools matters. Co-locating them within the same rack or node minimizes this overhead.

### Research Systems

**Splitwise** (Patel et al., ISCA 2024): Proposed disaggregation with a focus on cost optimization. Key insight: prefill and decode have different cost-optimal GPU configurations. Prefill benefits from fewer, more powerful GPUs; decode benefits from more, bandwidth-optimized GPUs. Mixed clusters (e.g., H100 for prefill, A100 for decode) can reduce total cost.

**DistServe** (Zhong et al., OSDI 2024): Introduced "prefill-decode disaggregation" with a placement algorithm that optimally assigns prefill and decode workloads to GPUs based on their compute-to-bandwidth ratios. Demonstrated 1.5-2x cost-efficiency improvement on production-like workloads. Also proposed "KV cache relay" to overlap transfer with computation.

**Mooncake** (Moonshot AI, 2024): A production disaggregated architecture using a KV cache-centric design. Prefill instances write KV caches to a distributed cache pool; decode instances read from it. This decouples the two phases entirely, with the cache pool acting as an intermediary.

## Why It Matters

1. **Eliminates decode latency interference**: Users never experience mid-response pauses caused by another request's prefill. This is critical for interactive applications where consistent token delivery rate directly affects user experience.
2. **Better hardware utilization**: Prefill GPUs run at high compute utilization; decode GPUs run at high memory bandwidth utilization. Neither pool wastes resources on work mismatched to its hardware profile.
3. **Cost-efficiency gains of 1.5-2x**: By matching hardware to workload characteristics, disaggregation reduces the total GPU-hours needed to serve the same traffic. This is especially pronounced for workloads with long inputs and short outputs (e.g., summarization, classification, extraction).
4. **Independent scaling**: Prefill and decode pools can be scaled independently based on workload mix. A sudden influx of long-prompt requests can be handled by adding prefill capacity without affecting decode throughput.
5. **Enables heterogeneous hardware**: Prefill pools can use the latest high-FLOPS GPUs while decode pools use older or cheaper GPUs optimized for bandwidth. This extends the useful life of hardware and reduces capital expenditure.

## Key Technical Details

- **Workload sensitivity**: Disaggregation benefits workloads with long inputs and short outputs most (e.g., summarization: 8K input, 200 output). For workloads with short inputs and long outputs (e.g., creative writing: 100 input, 4K output), the prefill phase is negligible and disaggregation adds transfer overhead with little benefit.
- **KV cache transfer overhead**: At inter-node bandwidth (50 GB/s), transferring a 1.3 GB KV cache adds ~26 ms latency. For latency-sensitive applications, this must be weighed against the interference-elimination benefit.
- **Chunked prefill alternative**: An intermediate approach between full disaggregation and no disaggregation. Chunked prefill breaks the input into smaller pieces and interleaves them with decode steps on the same GPU. This reduces interference without requiring KV cache transfer, but does not fully eliminate it or enable hardware specialization.
- **Scheduling complexity**: A disaggregated system needs a global scheduler that routes requests to prefill instances, manages KV cache transfer, and assigns sequences to decode instances. This adds orchestration complexity compared to monolithic serving.
- **Memory duplication risk**: If the KV cache exists on both the prefill GPU (before transfer) and the decode GPU (after transfer), total memory usage temporarily doubles. Efficient implementations pipeline the transfer and free prefill-side memory promptly.
- **Batch size dynamics**: Prefill instances can batch multiple requests' prefills together for higher compute utilization. Decode instances maintain their own continuous batching. The two batch sizes are optimized independently.

## Common Misconceptions

- **"Disaggregation always improves performance."** For workloads dominated by decode (short inputs, long outputs), the KV cache transfer overhead can outweigh the benefits. Disaggregation is most valuable for long-input workloads or mixed workloads where interference is severe.
- **"You need two completely separate GPU clusters."** Disaggregation can operate within a single node by dedicating some GPUs to prefill and others to decode, using NVLink for near-zero transfer overhead. Full cross-node disaggregation is one option but not the only one.
- **"Chunked prefill makes disaggregation unnecessary."** Chunked prefill mitigates interference but does not eliminate it, and it cannot specialize hardware. Disaggregation provides stronger isolation and enables heterogeneous GPU deployments. They address overlapping but distinct problems.
- **"The KV cache transfer is a bottleneck."** With NVLink (intra-node), transfer time is typically 1-5 ms for practical prompt lengths -- negligible. Even across InfiniBand, INT8 compressed KV transfer for a 4K-token prompt takes about 13 ms, which is small compared to the prefill computation itself (often 50-200 ms for long prompts).

## Connections to Other Concepts

- **Continuous Batching**: Continuous batching optimizes scheduling within a single GPU. Disaggregation extends this by optimizing scheduling across specialized GPU pools. Decode pools still run continuous batching internally.
- **PagedAttention**: Efficient KV cache management via PagedAttention is essential for both pools. The prefill pool allocates blocks during prefill; these block structures must be serialized for transfer and reconstructed on the decode pool.
- **KV Cache Compression**: Quantizing the KV cache before inter-pool transfer reduces bandwidth requirements. INT8 or INT4 KV cache compression directly halves or quarters the transfer time.
- **Prefix Caching**: In a disaggregated system, the decode pool can maintain a prefix cache. Requests with cached prefixes might skip the prefill pool entirely, reducing load on prefill instances and eliminating transfer latency.
- **Throughput vs. Latency**: Disaggregation explicitly decouples the throughput optimization (prefill) from the latency optimization (decode), allowing each to be tuned independently.

## Further Reading

- Patel et al., "Splitwise: Efficient Generative LLM Inference Using Phase Splitting" (ISCA 2024) -- Foundational paper analyzing the compute and memory profiles of prefill vs. decode and proposing hardware-aware disaggregation.
- Zhong et al., "DistServe: Disaggregating Prefill and Decoding for Goodput-optimized Large Language Model Serving" (OSDI 2024) -- Introduces the placement algorithm for optimal prefill/decode assignment with KV cache relay.
- Qin et al., "Mooncake: A KVCache-centric Disaggregated Architecture for LLM Serving" (2024) -- Production-scale disaggregated system with a distributed KV cache pool intermediary.
