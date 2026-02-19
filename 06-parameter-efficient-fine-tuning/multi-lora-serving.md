# S-LoRA / Multi-LoRA Serving

**One-Line Summary**: Multi-LoRA serving systems like S-LoRA enable thousands of LoRA adapters to be served simultaneously from a single shared base model, using unified memory management and custom CUDA kernels to maintain near-baseline throughput.

**Prerequisites**: LoRA (Low-Rank Adaptation), PagedAttention / vLLM, CUDA kernel programming, GPU memory hierarchy, KV caching

## What Is Multi-LoRA Serving?

Imagine a hotel with one large kitchen (the base model) that can prepare any cuisine by swapping in different recipe cards (LoRA adapters) for each guest's order. Rather than building a separate kitchen for every cuisine, the hotel uses one set of equipment and dynamically loads the right recipes as orders come in. Multi-LoRA serving works the same way: a single base model serves many fine-tuned variants by dynamically loading and swapping lightweight LoRA adapter weights.

*Recommended visual: S-LoRA unified paging architecture showing shared base model with dynamically loaded LoRA adapters — see [S-LoRA Paper (arXiv:2311.03285)](https://arxiv.org/abs/2311.03285)*


This matters because organizations increasingly fine-tune separate LoRA adapters for different tasks, customers, or domains. A company might have hundreds of adapters -- one per enterprise client, one per language, one per use case. The naive approach of loading each adapter as a separate model instance is wildly inefficient: a 7B model takes ~14GB in FP16, so serving 100 adapters would naively require 1.4TB of GPU memory. Since each LoRA adapter is only 10-50MB (0.1-0.5% of the base model), the shared-base approach reduces this to ~14GB base + ~5GB for all 100 adapters.

The challenge is engineering: how do you efficiently batch requests that target different adapters, manage adapter weights in GPU memory, and compute the LoRA additions without destroying throughput? Systems like S-LoRA, Punica, and LoRAX solve these problems with custom memory management, specialized CUDA kernels, and intelligent adapter scheduling.

## How It Works


*Recommended visual: Multi-LoRA batching diagram showing heterogeneous requests each using different adapters served from a single GPU — see [S-LoRA Paper Figure 1](https://arxiv.org/abs/2311.03285)*

### Unified Paging for LoRA Weights (S-LoRA)

S-LoRA extends the PagedAttention concept from vLLM to LoRA adapter weights. Just as PagedAttention manages KV cache memory in non-contiguous pages to avoid fragmentation, S-LoRA stores LoRA weight matrices (A and B) in a unified memory pool with page-level granularity:

```
Base model weights:  Fixed in GPU memory (shared by all requests)
LoRA adapter pool:   Paged memory manager
  - Adapter_1 (A1, B1): Pages [0x00, 0x01, 0x02]
  - Adapter_2 (A2, B2): Pages [0x03, 0x04]
  - Adapter_3 (A3, B3): Pages [0x05, 0x06, 0x07]
  ...
  - Adapter_N: Evicted to CPU (LRU policy)
```

When GPU memory fills up, least-recently-used adapters are evicted to CPU RAM and reloaded on demand. The paging system avoids memory fragmentation that would otherwise limit the number of concurrent adapters. The key insight is that adapter weights are small enough that CPU-to-GPU transfer latency (~1-5ms) is negligible compared to the inference latency of the base model itself.

### Custom CUDA Kernels: Batched SGMV

The core computational challenge is that different requests in a batch may use different LoRA adapters. Standard batched matrix multiplication assumes uniform weight matrices. The **SGMV (Segmented Gather Matrix-Vector)** kernel, introduced by Punica, solves this:

```
Standard batched GEMM:  Y = X @ W           (same W for all)
SGMV for multi-LoRA:    Y_i = X_i @ B_i @ A_i  (different A_i, B_i per request)
```

The SGMV kernel performs a segmented gather to collect the right LoRA weights for each request in the batch, then executes the low-rank multiplication efficiently using shared memory and warp-level primitives. This adds only ~2% overhead compared to serving the base model alone, because the LoRA computation (typically rank 8-64) is tiny compared to the base model's full-rank operations.

The mathematical operation for each request i in the batch is:

```
output_i = base_output_i + x_i @ A_i^T @ B_i^T * scaling_i
```

where A_i and B_i are the LoRA matrices for request i's adapter, and scaling_i = alpha/rank.

### Heterogeneous Batching and Scheduling

S-LoRA's request scheduler batches requests targeting different adapters together into a single batch, with metadata indicating which adapter each request uses. The scheduler balances several concerns:

- **Adapter locality**: Grouping requests for the same adapter improves cache efficiency
- **LRU eviction**: Tracking adapter access patterns to keep hot adapters in GPU memory
- **Admission control**: Limiting batch size to avoid excessive adapter loading overhead
- **Prefetch**: Predicting which adapters will be needed and loading them proactively
- **Priority management**: Ensuring high-priority adapters get preferential GPU memory residence

The scheduler also integrates with continuous batching, so requests can enter and leave the batch at different times without blocking on adapter loading.

### Production Systems: LoRAX and Beyond

LoRAX (from Predibase) is a production-grade framework built on these principles. It adds several critical features for production deployments:

- **Dynamic adapter loading**: Adapters are loaded from object storage (S3, GCS) on demand, with no server restart required
- **Tiered caching**: Three-level cache (GPU -> CPU -> disk/network) with intelligent prefetching
- **Adapter weight format optimization**: Supports multiple serialization formats and quantized adapters
- **Serving framework integration**: Compatible with TGI-style APIs and OpenAI-compatible endpoints
- **Multi-adapter composition**: Support for merging multiple LoRA adapters per request

LoRAX supports hot-swapping adapters at runtime, enabling use cases like A/B testing different fine-tunes without restarting the server.

## Why It Matters

1. **Massive cost reduction**: Serving 2000 adapters from one base model on a single GPU instead of deploying 2000 separate model instances reduces infrastructure costs by orders of magnitude.
2. **Multi-tenant SaaS**: Enables platforms to offer personalized models to each customer without dedicated GPU allocations per tenant.
3. **Rapid experimentation**: A/B testing between different fine-tuned variants becomes trivial -- just route requests to different adapter IDs without any deployment changes.
4. **Task-specific routing**: Combine multi-LoRA serving with a router to automatically select the best adapter (coding, math, creative writing) per query.
5. **Memory efficiency**: LoRA adapters are 0.1-0.5% of base model size, so thousands of specializations fit in the memory footprint of approximately one additional model copy.

## Key Technical Details

- S-LoRA serves up to 2000 concurrent adapters on a single A100 (80GB) with only 4% throughput degradation vs. base model serving
- S-LoRA achieves 30x higher throughput compared to naive adapter switching (load/unload per request)
- Punica's SGMV kernel adds ~2% latency overhead for LoRA computation in heterogeneous batches
- Typical LoRA adapter sizes: rank 8 = ~10MB, rank 16 = ~20MB, rank 64 = ~50MB (for a 7B model in FP16)
- Adapter loading latency from CPU to GPU: ~1-5ms for typical adapter sizes via PCIe
- LRU eviction policy keeps the top-K most frequently accessed adapters in GPU memory
- LoRAX supports adapters in multiple formats: HuggingFace PEFT, custom binary, quantized
- Maximum practical adapter count is limited by CPU memory (for evicted adapters) rather than GPU memory
- Heterogeneous batching works best when adapter distribution follows a power law (a few adapters get most traffic)
- S-LoRA's unified memory pool reserves ~10-20% of GPU memory for adapter weights, with the rest for KV cache and base model
- Adapter rank affects both size and compute: rank 64 adapters use ~8x more memory and compute than rank 8

## Common Misconceptions

- **"Each LoRA adapter needs its own GPU."** The entire point of multi-LoRA serving is that adapters are tiny and share the base model. Thousands of adapters can coexist on a single GPU with intelligent memory management.
- **"Batching requests for different adapters is inefficient."** The SGMV kernel makes heterogeneous batching nearly as efficient as homogeneous batching, with only ~2% overhead. The LoRA computation itself is a small fraction of total inference cost.
- **"You need to restart the server to add a new adapter."** Production systems like LoRAX support dynamic hot-loading of adapters at runtime -- upload a new adapter to storage and it becomes available immediately.
- **"Multi-LoRA serving changes the model outputs."** The outputs are mathematically identical to running each adapter independently. The system optimizations are purely about memory management and compute scheduling.
- **"All adapters must use the same rank."** S-LoRA and LoRAX support heterogeneous adapter ranks within the same serving instance, using the paging system to handle variable-size weight matrices.

## Connections to Other Concepts

- **LoRA**: Multi-LoRA serving is the deployment counterpart to LoRA training. The small adapter sizes that make LoRA parameter-efficient also make multi-tenant serving feasible.
- **PagedAttention / vLLM**: S-LoRA extends PagedAttention's memory management philosophy from KV caches to adapter weights, using the same non-contiguous paging approach.
- **Model Routing**: Multi-LoRA serving pairs naturally with routing systems that select the best adapter per query, enabling automatic task-specific specialization.
- **Quantization**: Base model quantization (GPTQ, AWQ) can be combined with LoRA serving to further reduce the memory footprint, fitting more adapters per GPU.
- **Continuous Batching**: Multi-LoRA scheduling integrates with continuous batching to handle variable-length requests and adapter heterogeneity simultaneously.
- **Mixture of Experts**: Conceptually similar to MoE where different experts activate for different inputs, but implemented at the adapter level with explicit routing rather than learned gating.

## Further Reading

- Sheng, Y., Cao, S., Li, D., Hooper, C., Lee, N., Yang, S., Chou, C., Zhu, B., Zheng, L., Keutzer, K., Gonzalez, J. E., & Stoica, I. (2024). "S-LoRA: Serving Thousands of Concurrent LoRA Adapters." MLSys 2024. arXiv:2311.03285.
- Chen, L.-A., Ye, Z., Wu, Y., Zhuo, D., Ceze, L., & Krishnamurthy, A. (2024). "Punica: Multi-Tenant LoRA Serving." MLSys 2024. arXiv:2310.18547.
- Predibase. (2024). "LoRAX: Multi-LoRA Inference Server." https://github.com/predibase/lorax.
- Hu, E. J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., Wang, L., & Chen, W. (2022). "LoRA: Low-Rank Adaptation of Large Language Models." ICLR 2022. arXiv:2106.09685.
- Dettmers, T., Pagnoni, A., Holtzman, A., & Zettlemoyer, L. (2023). "QLoRA: Efficient Finetuning of Quantized Language Models." NeurIPS 2023. arXiv:2305.14314.
