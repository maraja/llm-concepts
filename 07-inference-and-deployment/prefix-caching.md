# Prefix Caching

**One-Line Summary**: Prefix caching stores the computed KV cache states for shared prompt prefixes (system prompts, few-shot examples, RAG context) so that subsequent requests sharing the same prefix skip recomputation entirely, delivering up to 90% cost savings and 85% reduction in time-to-first-token.

**Prerequisites**: KV cache, PagedAttention, prefill phase, tokenization, autoregressive generation, system prompts and prompt engineering patterns.

## What Is Prefix Caching?

Imagine a law firm where every legal brief begins with the same 20-page boilerplate: firm name, jurisdiction, standard disclaimers, and regulatory citations. Without prefix caching, every attorney drafting a brief retyping those 20 pages from scratch before adding their unique case arguments. With prefix caching, the firm keeps a master copy of the boilerplate already formatted and typeset. Each new brief starts from page 21, and the attorney only writes the novel content. The time and cost savings are enormous when thousands of briefs are produced daily.

In LLM serving, the "boilerplate" is the system prompt, few-shot examples, or RAG-retrieved documents that appear identically at the beginning of many requests. During the prefill phase, the model processes these input tokens to compute key-value vectors for every attention layer. For a 4,000-token system prompt on a 70B model, this prefill computation is substantial -- potentially hundreds of milliseconds of GPU time. When thousands of requests per minute share the same system prompt, the system recomputes identical KV states over and over again.

Prefix caching eliminates this redundancy. The first request computes the KV cache for the shared prefix and stores it. Every subsequent request that begins with the same token sequence reuses the cached KV states and only computes the prefill for the unique suffix (the user's actual query). The savings scale with how much of the prompt is shared and how many requests benefit from the cache.

## How It Works

### Exact Token Matching

Prefix caching requires an **exact, token-level match** from the first token onward. Even a single different token breaks the prefix match at that point. This is because the KV cache at position N depends on all tokens from position 0 to N -- changing any earlier token invalidates all subsequent cached states.

```
Request 1: [SYS_PROMPT (2000 tokens)] + [User: "What is photosynthesis?"]
Request 2: [SYS_PROMPT (2000 tokens)] + [User: "Explain quantum computing"]
Request 3: [SYS_PROMPT (2000 tokens)] + [User: "Write a poem about rain"]

Without prefix caching:
  Each request prefills all ~2020 tokens → 3 full prefills

With prefix caching:
  Request 1: Prefill 2000 (system) + 20 (user) → cache 2000-token prefix
  Request 2: Reuse 2000 cached tokens, prefill only ~20 new tokens
  Request 3: Reuse 2000 cached tokens, prefill only ~20 new tokens
  → 1 full prefill + 2 tiny prefills (99% reduction for requests 2 & 3)
```

### RadixAttention (SGLang)

SGLang implements prefix caching through RadixAttention, which organizes all cached KV blocks in a radix tree (trie) indexed by token sequences. The radix tree provides:

- **Automatic prefix matching**: When a new request arrives, the system traverses the tree from the root, following the token sequence. The traversal stops at the longest matching prefix, and only the remaining tokens need prefill computation.
- **Fine-grained sharing**: Different requests can share varying prefix lengths. Request A might match 2,000 tokens, while Request B (with a slightly different few-shot example) matches only 1,500.
- **LRU eviction**: When GPU memory is full, the least recently used leaf nodes are evicted first, preserving frequently accessed prefixes.

```
Radix Tree Structure:

            [ROOT]
           /      \
    [System A]   [System B]
    (2000 tok)   (1500 tok)
     /    \          |
  [RAG    [RAG    [Few-shot
   Doc1]   Doc2]   Examples]
  (800t)  (600t)   (400t)
```

### API-Level Prefix Caching

Major API providers have productionized prefix caching:

- **Anthropic**: Offers explicit cache control via `cache_control` breakpoints in the API. Cached input tokens are billed at a reduced rate (typically 90% discount). Cache has a minimum prefix length (1024 tokens for Claude) and a TTL (time-to-live) of 5 minutes with automatic extension on cache hits.
- **OpenAI**: Automatic prefix caching for prompts longer than 1024 tokens. Cached tokens are billed at 50% of the standard input rate. Caching happens transparently without API changes.
- **Google (Gemini)**: Offers "context caching" as an explicit API feature where users create a named cache object with a configurable TTL.

### Implementation with PagedAttention

At the block level, prefix caching maps naturally onto PagedAttention. Cached KV blocks for a shared prefix are marked as read-only in the block pool. New requests' block tables point to these shared physical blocks for the prefix portion, then allocate fresh blocks for their unique suffixes. Copy-on-write semantics ensure correctness if any block would need modification (though in practice, prefix blocks are never modified).

## Why It Matters

1. **Massive cost reduction**: In production workloads where system prompts or RAG contexts are repeated across thousands of requests, prefix caching reduces input token costs by 50-90%, directly impacting the economics of LLM deployment.
2. **Time-to-first-token (TTFT) reduction**: Skipping the prefill of a 4,000-token system prompt can reduce TTFT by 85% or more, dramatically improving perceived responsiveness for users.
3. **Enables longer system prompts**: Without prefix caching, long system prompts (detailed instructions, many few-shot examples, large RAG contexts) are prohibitively expensive at scale. Prefix caching makes it economical to use rich, detailed prompts.
4. **Multi-tenant efficiency**: In platforms serving multiple applications, each with its own system prompt, prefix caching allows the serving infrastructure to maintain hot caches for the most common prompts across all tenants.
5. **Synergy with RAG architectures**: RAG systems often prepend the same retrieved documents to multiple follow-up queries. Prefix caching ensures the document processing cost is paid only once.

## Key Technical Details

- **Minimum prefix length**: Most implementations require a minimum prefix length (e.g., 1024 tokens for Anthropic's API) to ensure the caching overhead is worthwhile.
- **Cache granularity**: Block-level caching (PagedAttention blocks of 16-32 tokens) means partial prefix matches are possible down to block boundaries, not just all-or-nothing.
- **Cache invalidation**: Any change to the prefix tokens invalidates the cache from that point forward. This includes changes in tokenization (different model versions may tokenize the same text differently).
- **Memory cost**: Cached KV blocks consume GPU memory. A 4,000-token prefix for a 70B model at FP16 requires approximately 1.25 GB of KV cache memory (across all layers and heads). Systems must balance cache size against active generation memory.
- **TTL and eviction**: Caches have finite lifetimes. Anthropic uses a 5-minute TTL that resets on each hit. Self-hosted systems typically use LRU eviction based on GPU memory pressure.
- **Multi-turn conversations**: Each turn extends the prefix. Turn 1's entire context (system prompt + user message + assistant response) becomes the prefix for Turn 2's prefill, enabling progressive caching across a conversation.
- **Prompt structure matters**: To maximize cache hits, prompts should be structured with stable content (system prompt, instructions) first and variable content (user query) last.

## Common Misconceptions

- **"Prefix caching works with approximate matches."** It requires exact token-level matching from position 0. Even inserting a timestamp or request ID at the beginning of the prompt will break the prefix match entirely. Prompt design must place variable content at the end.
- **"Prefix caching helps with output generation speed."** It accelerates only the prefill phase (processing input tokens). The decode phase (generating output tokens one at a time) is unaffected because each new output token's KV vectors are unique to the sequence.
- **"You need to explicitly manage the cache."** In many systems (vLLM, SGLang, OpenAI's API), prefix caching is automatic and transparent. The system detects shared prefixes without user intervention. Anthropic's API is an exception, offering explicit cache control for finer-grained management.
- **"Prefix caching is only useful for system prompts."** Any shared prefix benefits -- few-shot examples, RAG documents, conversation history, shared preambles in batch processing. The technique is general to any repeated prefix pattern.

## Connections to Other Concepts

- **KV Cache**: Prefix caching is fundamentally about reusing the KV cache. Understanding what the KV cache stores (key and value projections at every layer for every token) and how it grows is essential context.
- **PagedAttention**: Block-level memory management enables efficient prefix sharing through shared block references and copy-on-write semantics. Without PagedAttention, prefix caching would require large contiguous memory reservations.
- **Flash Attention**: Flash Attention computes attention efficiently during the prefill phase. Prefix caching eliminates the need for that computation entirely for cached tokens -- a complementary optimization.
- **Continuous Batching**: New requests admitted via continuous batching benefit immediately from prefix cache hits, making their prefill phase nearly instantaneous and allowing faster batch admission.
- **Throughput vs. Latency**: Prefix caching primarily reduces latency (TTFT) and cost for individual requests. At the system level, reduced prefill compute frees GPU cycles for more decode throughput.

## Diagrams and Visualizations

*Recommended visual: Prefix caching showing shared system prompt KV cache reused across multiple requests — see [SGLang RadixAttention Paper (arXiv:2312.07104)](https://arxiv.org/abs/2312.07104)*

*Recommended visual: Radix tree data structure indexing KV cache blocks by token content for automatic prefix matching — see [SGLang Documentation](https://sgl-project.github.io/)*

## Further Reading

- Zheng et al., "SGLang: Efficient Execution of Structured Language Model Programs" (2024) -- Introduces RadixAttention with its radix-tree-based prefix caching and LRU eviction strategy.
- Anthropic, "Prompt Caching Documentation" (2024) -- Practical guide to using explicit cache control in production API deployments with pricing details.
- Gim et al., "Prompt Cache: Modular Attention Reuse for Low-Latency Inference" (2024) -- Academic treatment of prompt-level KV cache reuse with analysis of cache hit rates across different workload patterns.
