# Continuous Batching

**One-Line Summary**: Continuous batching (also called iteration-level or in-flight batching) inserts new requests and retires completed sequences at every decoding step rather than waiting for an entire batch to finish, eliminating idle GPU cycles and achieving 10-23x higher throughput than static batching.

**Prerequisites**: Autoregressive generation, KV cache, PagedAttention, the distinction between prefill and decode phases, GPU utilization and throughput vs. latency trade-offs.

## What Is Continuous Batching?

Picture a restaurant where the kitchen refuses to take any new orders until every table from the current seating has finished eating. Table 1 finishes in 15 minutes, but table 8 ordered a seven-course meal that takes 90 minutes. Tables 2 through 7 have long since paid and left, their places sitting empty, while a line of hungry customers grows outside the door. No new food is being prepared for those empty seats -- the kitchen simply waits. This is static batching.

Now imagine the restaurant adopts a rolling policy: the moment a table clears, a new party is seated immediately and their order goes into the kitchen. The kitchen is continuously serving food, never idle, and the line outside the door moves steadily. This is continuous batching, and it is one of the highest-impact optimizations in modern LLM serving.

In static batching, a group of requests is assembled into a batch, processed together until every sequence in the batch has finished generating (by producing an end-of-sequence token or reaching the maximum length), and only then is the next batch admitted. Because sequences vary dramatically in output length -- one might generate 10 tokens, another 500 -- the GPU sits idle for finished sequences while it waits for the longest one. With a batch of 32 sequences where most finish at 50 tokens but one runs to 500, the GPU is doing useful work for only a fraction of those 32 slots during the tail of generation. Continuous batching eliminates this waste entirely.

## How It Works

### Iteration-Level Scheduling

In continuous batching, the scheduler operates at the granularity of a single decoding step (one token of generation). After each step, the scheduler:

1. **Retires** any sequence that has produced an end-of-sequence token or hit its length limit. Its KV cache memory is freed immediately.
2. **Admits** new requests from the waiting queue, up to the available memory and batch-size budget. Each new request undergoes its prefill phase (processing its input tokens).
3. **Continues** decoding for all remaining active sequences.

```
Static Batching Timeline (batch size = 4):
Step:   1  2  3  4  5  6  7  8  9  10
Seq A: [=  =  =  DONE  .  .  .  .  .  .]
Seq B: [=  =  =  =  =  =  DONE  .  .  .]
Seq C: [=  =  DONE  .  .  .  .  .  .  .]
Seq D: [=  =  =  =  =  =  =  =  =  DONE]
                 ↑ Seq A,C done but slots idle until step 10

Continuous Batching Timeline:
Step:   1  2  3  4  5  6  7  8  9  10
Seq A: [=  =  =  DONE]
Seq C: [=  =  DONE]
Seq E:          [=  =  =  =  DONE]         ← fills C's slot
Seq F:             [=  =  DONE]             ← fills A's slot
Seq B: [=  =  =  =  =  =  DONE]
Seq G:                      [=  =  =  DONE] ← fills B's slot
Seq D: [=  =  =  =  =  =  =  =  =  DONE]
                 ↑ Every slot is always doing useful work
```

### Prefill-Decode Interleaving

A subtle challenge is that newly admitted requests need their prefill phase (processing potentially thousands of input tokens in parallel), which is compute-heavy, while existing sequences are in the decode phase (generating one token per step), which is memory-bandwidth-heavy. Continuous batching systems handle this through several strategies:

- **Chunked prefill**: Break the new request's prompt into chunks (e.g., 512 tokens at a time) and interleave prefill chunks with decode steps. This prevents a single large prefill from stalling all in-flight decode operations.
- **Piggyback prefill**: Process the new request's prefill alongside the existing batch's decode step in the same GPU kernel call, exploiting the fact that decode underutilizes compute.
- **Separate prefill scheduling**: Some systems prioritize decode operations and schedule prefills during periods of lower batch occupancy.

### Memory Management Integration

Continuous batching works hand-in-hand with PagedAttention. When a sequence finishes, its KV cache blocks are returned to the free pool instantly, and those blocks are available for the new sequence's KV cache. Without dynamic memory management, admitting a new request mid-batch would require pre-reserved contiguous memory -- which defeats the purpose.

```python
# Simplified continuous batching loop (pseudocode)
while requests_pending or active_sequences:
    # Retire finished sequences
    for seq in active_sequences:
        if seq.is_done():
            free_kv_blocks(seq)
            active_sequences.remove(seq)
            send_response(seq)

    # Admit new requests up to memory budget
    while requests_pending and memory_available():
        new_seq = request_queue.pop()
        allocate_kv_blocks(new_seq)
        run_prefill(new_seq)
        active_sequences.add(new_seq)

    # One decode step for all active sequences
    for seq in active_sequences:
        next_token = decode_step(seq)
        seq.append(next_token)
```

## Why It Matters

1. **Throughput transformation**: Continuous batching delivers 10-23x higher throughput compared to static batching on the same hardware, as measured by the original Orca paper and confirmed by production deployments. This is one of the single largest efficiency gains in the LLM inference stack.
2. **Cost reduction**: Serving the same traffic with 10-20x fewer GPU-hours translates directly to an order-of-magnitude reduction in infrastructure cost.
3. **Latency improvement for short requests**: In static batching, a short request that lands in a batch with long sequences must wait for the entire batch to complete. In continuous batching, the short request finishes and returns immediately.
4. **Industry standard**: Continuous batching is implemented in every major LLM serving framework -- vLLM, TensorRT-LLM, Text Generation Inference (TGI), SGLang, and Triton Inference Server. Deploying without it is effectively leaving 90% of your GPU budget on the table.
5. **Enables SLA compliance**: By decoupling individual request latencies from the behavior of other requests in the batch, continuous batching makes it practical to offer latency guarantees (e.g., p99 TTFT < 500ms).

## Key Technical Details

- **Batch size is dynamic**: Unlike static batching where batch size is fixed, continuous batching adjusts the number of active sequences at every step based on available memory. The effective batch size fluctuates continuously.
- **Maximum batch size**: Still bounded by GPU memory. With a 70B model on an 80GB A100, the KV cache budget limits active sequences to roughly 40-100 depending on sequence lengths and quantization.
- **Scheduling policies**: Common approaches include first-come-first-served (FCFS), shortest-job-first (SJF), and priority-based scheduling. FCFS is standard; SJF can reduce average latency but requires output length prediction.
- **Prefill priority**: Systems must balance prefill latency (time-to-first-token for new requests) against decode throughput (tokens-per-second for in-flight requests). Aggressive prefill admission can cause decode latency spikes.
- **Token budget per step**: Some systems cap the total number of tokens processed per step (prefill + decode) to maintain consistent step latency, which is critical for real-time applications.
- **Chunked prefill granularity**: vLLM uses configurable chunk sizes (default 512 tokens). Smaller chunks give smoother latency but increase scheduling overhead.

## Common Misconceptions

- **"Continuous batching is the same as dynamic batching."** Dynamic batching (as in Triton Inference Server) groups incoming requests into batches at the request level, but still processes each batch to completion. Continuous batching operates at the token-generation (iteration) level, which is a fundamentally finer granularity.
- **"Continuous batching increases individual request latency."** For any given request, latency is the same or lower compared to static batching. The request is never forced to wait for unrelated longer sequences to complete.
- **"You just need a larger batch size to get the same throughput."** Increasing static batch size helps utilization but cannot solve the fundamental problem: variance in output length means some slots are always idle. Continuous batching solves the structural waste, not just the scale.
- **"Continuous batching is complex to implement."** While the scheduling logic adds complexity, the core concept is straightforward, and mature implementations exist in all major serving frameworks. The engineering challenge is primarily in efficient KV cache management, which PagedAttention handles.

## Connections to Other Concepts

- **PagedAttention**: The dynamic memory allocation of PagedAttention is what makes continuous batching practical. Without it, admitting and retiring sequences mid-batch would require expensive memory defragmentation.
- **Prefill-Decode Disaggregation**: An advanced extension of the insight behind continuous batching -- if prefill and decode have different compute profiles, why not run them on separate hardware entirely?
- **Throughput vs. Latency**: Continuous batching primarily optimizes throughput (total tokens per second across all requests) while also improving tail latency for short requests. The trade-off between prefill admission and decode smoothness is a key tuning dimension.
- **Speculative Decoding**: Speculative decoding generates variable numbers of tokens per step (depending on acceptance), requiring the scheduler to handle variable-length advances per sequence -- a natural extension of continuous batching's flexibility.
- **Model Serving Frameworks**: Continuous batching is the core scheduling innovation that differentiates modern LLM-specific serving systems (vLLM, TGI) from general-purpose model serving (basic Triton, Flask + PyTorch).

## Further Reading

- Yu et al., "Orca: A Distributed Serving System for Transformer-Based Generative Models" (2022) -- The paper that introduced iteration-level scheduling (continuous batching) and demonstrated order-of-magnitude throughput improvements.
- Kwon et al., "Efficient Memory Management for Large Language Model Serving with PagedAttention" (2023) -- PagedAttention and vLLM, which made continuous batching practical with dynamic KV cache management.
- NVIDIA, "TensorRT-LLM In-Flight Batching" (2024) -- NVIDIA's production implementation of continuous batching with optimized CUDA kernels.
