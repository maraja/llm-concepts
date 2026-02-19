# Model Serving Frameworks

**One-Line Summary**: Model serving frameworks handle the complex orchestration of loading LLM weights onto GPUs, managing memory, batching requests, and delivering generated tokens to users -- and the choice of framework can mean a 10-23x difference in throughput for the same hardware.

**Prerequisites**: GPU memory basics (HBM), KV cache concept, batching, HTTP APIs and server architecture, basic understanding of quantization formats, the distinction between latency and throughput.

## What Is Model Serving?

Running an LLM is not as simple as loading weights and calling a forward pass. A production serving system must handle dozens of concurrent users with different prompt lengths and generation requirements, keep GPUs maximally utilized, manage gigabytes of dynamically growing KV caches, and deliver tokens back to users as they are generated.

*Recommended visual: vLLM architecture diagram showing PagedAttention, continuous batching, and request scheduling — see [vLLM Documentation](https://docs.vllm.ai/en/latest/)*


Think of a restaurant kitchen. The chef (GPU) can cook any dish (process any request), but the restaurant needs a system for taking orders, managing the kitchen queue, ensuring ingredients (memory) are allocated efficiently, and delivering plates as they are ready -- not waiting until every table's order is complete. Model serving frameworks are this restaurant management system.

## How It Works


*Recommended visual: Comparison of LLM serving frameworks (vLLM, TensorRT-LLM, SGLang) throughput benchmarks — see [vLLM Paper (arXiv:2309.06180)](https://arxiv.org/abs/2309.06180)*

### The Fundamental Challenge: Continuous Batching

Traditional static batching waits until a batch of requests is collected, processes them together, and returns all results when the *slowest* request finishes. This is terribly wasteful: a request generating 10 tokens waits for a request generating 500 tokens.

**Continuous batching** (also called iteration-level batching or inflight batching) solves this:

1. Requests enter a waiting queue as they arrive.
2. At each decode iteration, the scheduler checks for completed requests and newly arrived requests.
3. Completed requests are removed from the batch and their GPU memory is freed.
4. New requests are inserted into the batch if memory is available.
5. One forward pass is executed for all active requests simultaneously.

This means the batch composition changes at every iteration. No request waits for others. The GPU processes the maximum number of requests at all times.

The impact is dramatic: continuous batching improves throughput by **10-23x** over static batching in realistic scenarios, according to benchmarks by the vLLM team.

### Framework Comparison

#### vLLM

**Core innovation**: PagedAttention for efficient KV cache memory management.

- **PagedAttention**: Manages KV cache as non-contiguous memory pages, reducing waste from ~60-80% to under 4%. This is the single biggest throughput improvement.
- **Continuous batching**: Full support with iteration-level scheduling.
- **Quantization support**: AWQ, GPTQ, FP8, and more.
- **Speculative decoding**: Supported with configurable draft models.
- **Tensor parallelism**: Distributes large models across multiple GPUs.
- **OpenAI-compatible API**: Drop-in replacement for OpenAI API endpoints.
- **Best for**: General-purpose GPU serving, high throughput, research-friendly Python codebase.
- **Limitations**: Not as optimized as TensorRT-LLM for maximum single-request latency on NVIDIA hardware.

#### TGI (Text Generation Inference)

**Core innovation**: Tight integration with the Hugging Face ecosystem.

- **Rust-based server**: High-performance HTTP server with gRPC support.
- **Flash Attention**: Built-in for all supported architectures.
- **Continuous batching**: Full support.
- **Quantization**: GPTQ, AWQ, bitsandbytes, EETQ, and FP8.
- **Watermarking**: Built-in support for text watermarking.
- **Guidance/grammar support**: Constrained generation with JSON schemas or regular expressions.
- **Best for**: Teams already in the Hugging Face ecosystem, production deployments needing structured output.
- **Limitations**: Smaller community than vLLM, fewer cutting-edge optimization techniques.

#### TensorRT-LLM

**Core innovation**: Maximum performance on NVIDIA GPUs through deep hardware-specific optimization.

- **Compiled execution**: Models are compiled into optimized TensorRT engines with fused kernels.
- **FP8 support**: First-class support for Hopper GPU FP8, achieving near-2x speedup over FP16.
- **Inflight batching**: NVIDIA's continuous batching implementation.
- **KV cache quantization**: INT8 KV cache support for memory efficiency.
- **Multi-GPU**: Tensor and pipeline parallelism across multiple GPUs and nodes.
- **Speculative decoding**: Supported with various draft model configurations.
- **Best for**: Maximum throughput and minimum latency on NVIDIA hardware, large-scale production deployments.
- **Limitations**: Complex setup, NVIDIA-only, less flexible than Python-based frameworks, longer iteration cycle for new model support.

#### Ollama / llama.cpp

**Core innovation**: Making LLMs accessible on consumer hardware, especially Apple Silicon.

- **llama.cpp**: The foundational C/C++ inference engine. Supports CPU, Apple Metal, CUDA, and Vulkan.
- **GGUF format**: Flexible quantization format supporting many bit widths (Q2 through Q8, with k-quant variants).
- **Ollama**: A user-friendly wrapper around llama.cpp with a model registry, automatic downloads, and a simple CLI/API.
- **Apple Silicon optimization**: Excellent support for unified memory on M-series chips.
- **Best for**: Local inference, privacy-sensitive applications, hobbyists, development/testing.
- **Limitations**: Lower throughput than GPU-optimized solutions, limited multi-user serving capabilities, no continuous batching (though llama.cpp server has basic batching).

#### ONNX Runtime

**Core innovation**: Cross-platform inference with hardware abstraction.

- **ONNX format**: An open model interchange format supported by multiple hardware backends.
- **Execution providers**: CPU, CUDA, TensorRT, DirectML (Windows GPU), CoreML (Apple), OpenVINO (Intel), and more.
- **Quantization**: INT8 and INT4 quantization with the ONNX quantization toolkit.
- **Best for**: Cross-platform deployment, edge devices, scenarios requiring hardware portability.
- **Limitations**: Not specialized for autoregressive LLM serving -- lacks native continuous batching and PagedAttention. Better suited for encoder models and non-generative tasks.

### Performance Comparison Summary

| Framework | Throughput | Latency | Ease of Use | Hardware |
|-----------|-----------|---------|-------------|----------|
| vLLM | Very High | Good | Easy | NVIDIA, AMD |
| TensorRT-LLM | Highest | Best | Hard | NVIDIA only |
| TGI | High | Good | Medium | NVIDIA, AMD |
| Ollama/llama.cpp | Low-Medium | Good (local) | Very Easy | All (CPU, GPU, Metal) |
| ONNX Runtime | Medium | Medium | Medium | All platforms |

## Why It Matters

The serving framework is often the single biggest lever for inference cost reduction. Choosing vLLM with PagedAttention over naive static batching can reduce your GPU cost by 10-20x at the same throughput level. For organizations spending millions on inference compute, this translates directly to millions saved.

The framework also determines:
- **What hardware you can use** (NVIDIA-only vs. cross-platform).
- **What quantization formats are available** (GPTQ, AWQ, GGUF, FP8).
- **How quickly you can adopt new models** (Python frameworks iterate faster).
- **Your operational complexity** (Ollama is one command; TensorRT-LLM requires a build pipeline).

## Key Technical Details

- **Prefill-decode disaggregation**: Some advanced deployments use separate GPU pools for the prefill phase (compute-bound, benefits from high FLOPS) and decode phase (memory-bandwidth-bound, benefits from high memory bandwidth). Frameworks like TensorRT-LLM and vLLM are adding support for this pattern.
- **Prefix caching**: When multiple requests share the same system prompt, the KV cache for that prefix can be computed once and shared. vLLM calls this "automatic prefix caching."
- **Chunked prefill**: Long prompts can be processed in chunks interleaved with decode steps for other requests, preventing a single long prompt from blocking the entire batch.
- **Multi-LoRA serving**: vLLM and TGI can serve multiple LoRA adapters on top of a single base model, switching between them per-request without loading separate model copies.

## Common Misconceptions

- **"Just load the model and call generate()."** This works for single-user testing but is 10-20x less efficient than proper serving for production workloads. The overhead of a serving framework pays for itself immediately.
- **"The fastest framework is always the best choice."** TensorRT-LLM may achieve the highest throughput, but its complexity, build times, and NVIDIA lock-in make it the wrong choice for many teams. Engineering velocity matters.
- **"You need a datacenter GPU to serve LLMs."** Ollama and llama.cpp enable running quantized 7B-13B models on laptops and desktops, which is sufficient for many use cases.
- **"All serving frameworks use the same optimizations."** There are significant differences in memory management, kernel optimization, and scheduling that produce measurable performance gaps on identical hardware.

## Connections to Other Concepts

- **KV Cache**: Every serving framework's performance depends critically on how it manages KV cache memory. PagedAttention (vLLM) is the current gold standard.
- **Quantization**: Each framework supports different quantization formats and methods. Your quantization choice often dictates your framework choice and vice versa.
- **Flash Attention**: All major serving frameworks integrate Flash Attention as a foundational optimization. It is no longer optional.
- **Throughput vs. Latency**: Serving frameworks provide the knobs (batch size, scheduling policy, prefill chunking) to navigate this trade-off.
- **Speculative Decoding**: Framework support for speculative decoding is still maturing, with vLLM and TensorRT-LLM leading.

## Further Reading

1. **"Efficient Memory Management for Large Language Model Serving with PagedAttention"** (Kwon et al., 2023) -- The vLLM paper that introduced PagedAttention and demonstrated the throughput impact of intelligent KV cache memory management.
2. **"TensorRT-LLM: A Comprehensive Guide"** (NVIDIA, 2024) -- NVIDIA's documentation and benchmarks for their optimized serving solution.
3. **"Orca: A Distributed Serving System for Transformer-Based Generative Models"** (Yu et al., 2022) -- The paper that introduced iteration-level (continuous) batching, which all modern serving frameworks now implement.
