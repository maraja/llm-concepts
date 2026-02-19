# Prompt Compression / LLMLingua

**One-Line Summary**: Prompt compression reduces input token count while preserving semantic meaning, using perplexity-based importance scoring or trained classifiers to cut costs by up to 75% and accelerate prefill by 2-4x.

**Prerequisites**: Tokenization, perplexity, language model inference costs, retrieval-augmented generation (RAG), knowledge distillation

## What Is Prompt Compression?

Think of prompt compression like creating a telegram from a long letter. In the early 20th century, telegrams charged by the word, so people learned to strip out filler while keeping the essential meaning intact: "Arriving Tuesday 3PM, bring car" instead of "I wanted to let you know that I will be arriving on Tuesday at 3 PM, and it would be wonderful if you could bring the car to pick me up." Prompt compression does the same thing for LLM inputs -- it removes tokens that contribute little to the model's understanding while preserving the information the model actually needs to generate accurate responses.

This matters because LLM API costs scale linearly with input token count, and prefill latency scales quadratically with sequence length due to attention computation. A 10,000-token prompt that can be compressed to 2,500 tokens costs 75% less and processes roughly 2-4x faster during the prefill phase. For applications like RAG pipelines that routinely stuff multiple retrieved documents into context, compression can be the difference between a viable product and a cost-prohibitive one.

The field has evolved rapidly from simple heuristic approaches (removing stop words, truncating documents) to sophisticated learned methods. The LLMLingua family from Microsoft Research represents the state of the art, using small language models to identify which tokens carry the most information and which can be safely dropped without degrading downstream task performance.

## How It Works

### Perplexity-Based Token Importance (LLMLingua)

The original LLMLingua framework operates on a key insight: tokens that a small language model finds surprising (high perplexity) are likely the most informative and should be retained. The system has three components:

1. **Budget Controller**: Allocates the compression budget across different parts of the prompt (instructions, demonstrations, questions) based on their relative importance. Instructions typically get minimal compression (0-20% removal), while retrieved documents get aggressive compression (50-80% removal). The budget controller uses coarse-grained perplexity at the demonstration or segment level to determine allocation.

2. **Iterative Token-Level Compression**: A small language model (e.g., LLaMA-2 7B) computes perplexity for each token in the prompt. Tokens with low perplexity (highly predictable given context) are removed first, as the target LLM can likely reconstruct them. This process runs iteratively, recalculating perplexity after each round of removal to account for the changed context.

3. **Distribution Alignment**: A coarse-grained alignment step between the small compressor LM and the target LLM ensures that the small model's perplexity scores are meaningful proxies for what the large model would find informative. This is implemented via a temperature scaling calibration on the compressor model's output distribution.

```
Original:  "The capital city of France, which is known for the Eiffel Tower, is Paris."
Compressed: "capital France, known Eiffel Tower, Paris."
Compression: 14 tokens -> 7 tokens (2x)
```

The iterative nature is important: after removing some tokens, the perplexity of remaining tokens changes. A token that was predictable in the original context may become informative once surrounding tokens are removed.

### LongLLMLingua: Question-Aware Compression for RAG

LongLLMLingua extends the framework specifically for retrieval-augmented generation scenarios with long contexts. Key innovations include:

- **Question-aware compression**: Token importance is conditioned on the user's question, not just the local context. Tokens relevant to answering the question are preferentially retained even if they have low standalone perplexity. This is computed as the contrastive perplexity: the difference in perplexity with and without the question as a prefix.
- **Document reordering**: Retrieved documents are reordered by relevance before compression, placing the most important information in positions where LLMs attend most effectively (beginning and end of context), mitigating the "lost in the middle" problem.
- **Dynamic compression ratios**: Different documents receive different compression levels based on their relevance to the query, rather than applying a uniform compression ratio. A highly relevant document might keep 80% of tokens while a marginally relevant one keeps only 20%.
- **Subsequence recovery**: A post-processing step that maps the compressed output back to the original document, recovering exact spans for citation and attribution.

### LLMLingua-2: Classifier-Based Compression

LLMLingua-2 takes a fundamentally different approach. Instead of using perplexity from an autoregressive LM, it trains a lightweight token classifier to make binary keep/drop decisions for each token:

1. **Data generation**: GPT-4 is used to generate reference compressed texts from original prompts, creating supervised training labels. Each token in the original text gets a binary label: keep or drop.
2. **Token classification**: A small transformer encoder (e.g., XLM-RoBERTa or mBERT, ~560M parameters) is fine-tuned to predict for each token whether it should be kept or dropped.
3. **Inference**: The trained classifier runs a single forward pass over the prompt, producing keep/drop probabilities for all tokens simultaneously. A threshold controls the compression ratio.

This approach is 3-6x faster than LLMLingua v1 at compression time because it avoids iterative perplexity computation, uses a smaller encoder-only model, and processes all tokens in a single pass rather than autoregressively.

### Selective Context

An alternative approach, Selective Context, uses self-information to score token importance:

```
I(token) = -log P(token | context)
```

Tokens, phrases, or sentences with low self-information are removed. This is simpler than LLMLingua but operates at a coarser granularity (phrase or sentence level rather than token level), which limits its compression ratio at equivalent quality.

### Practical Pipeline Integration

In a typical production RAG pipeline, prompt compression sits between the retrieval stage and the LLM generation stage:

```
User query -> Retriever -> Top-K documents -> Prompt Compression -> LLM -> Response
```

The compression step receives the concatenated retrieved documents and the user query, applies question-aware compression to selectively retain relevant information, and outputs a compressed prompt that preserves the essential content for the LLM. The compression model runs on CPU or a small GPU, adding 50-500ms of latency but saving 2-4x on the subsequent LLM inference, yielding net latency reduction for prompts above ~2000 tokens.

Compression ratio can be dynamically adjusted based on the input length relative to the target model's context window. If the retrieved documents fit comfortably within the context window, light compression (2x) maximizes quality. If they exceed the window, aggressive compression (5-10x) is preferable to truncation.

## Why It Matters

1. **Direct cost reduction**: At 4x compression, input token costs drop by 75%. For high-volume applications processing millions of requests daily, this translates to substantial savings on API bills.
2. **Latency improvement**: Shorter prompts reduce prefill computation time by 2-4x due to the quadratic scaling of attention, directly improving user-perceived response times.
3. **Context window efficiency**: Compression allows fitting more information into fixed context windows, effectively expanding the model's usable context without architectural changes.
4. **RAG pipeline optimization**: Retrieved documents often contain significant amounts of irrelevant text. Compression selectively retains query-relevant information, improving both cost and answer quality.
5. **Edge deployment enablement**: Compressed prompts reduce memory and compute requirements, making it more feasible to deploy LLM applications on resource-constrained devices.

## Key Technical Details

- LLMLingua achieves up to 20x compression with 90%+ performance retention on benchmarks like GSM8K, BBH, and ShareGPT
- LLMLingua-2 compression speed: 3-6x faster than LLMLingua v1, processes ~10K tokens/second on a single GPU
- Selective Context uses self-information: I(token) = -log P(token | context) to score token importance at the phrase or sentence level
- Compression ratios of 2-5x are typical for minimal quality degradation; 10-20x is achievable with moderate quality loss
- LLMLingua uses LLaMA-2 7B as the compressor model; LLMLingua-2 uses XLM-RoBERTa-large (~560M params)
- Prefill latency reduction is roughly proportional to compression ratio for moderate-length prompts but sublinear for very long prompts due to attention's quadratic scaling
- Token-level compression preserves more information than sentence-level removal at the same compression ratio
- Compressed prompts may appear ungrammatical to humans but remain fully interpretable to LLMs
- LLMLingua-2 supports multilingual compression due to its XLM-RoBERTa backbone
- Compression overhead: LLMLingua ~100-500ms for 10K tokens, LLMLingua-2 ~50-100ms for 10K tokens

## Common Misconceptions

- **"Compressed prompts must be grammatically correct."** LLMs process tokens, not grammar. A compressed prompt like "capital France known Eiffel Tower Paris" is perfectly interpretable to a model even though it reads oddly to humans. The model reconstructs the implicit relationships from its training.
- **"Compression always degrades output quality."** At moderate compression ratios (2-4x), many benchmarks show negligible quality loss because the removed tokens were genuinely redundant. Quality degradation becomes significant only at aggressive compression ratios (10x+).
- **"You can just truncate the prompt instead."** Truncation removes contiguous blocks of text, often discarding critical information that appears later in the prompt. Learned compression selectively removes low-information tokens throughout, preserving the semantic skeleton of the entire input.
- **"Prompt compression is the same as text summarization."** Summarization produces coherent natural language summaries. Prompt compression produces token sequences optimized for LLM consumption, which may not be readable by humans but preserve the information the model needs.
- **"The compression model itself is expensive to run."** LLMLingua-2 uses a ~560M parameter encoder model that processes 10K tokens in ~50-100ms. This overhead is negligible compared to the savings from reduced LLM inference time on compressed prompts.

## Connections to Other Concepts

- **Retrieval-Augmented Generation (RAG)**: Prompt compression is especially valuable in RAG pipelines where multiple retrieved documents are concatenated into the prompt, often containing redundant or irrelevant passages.
- **KV Cache Compression**: Prompt compression reduces tokens before they enter the model; KV cache compression reduces the memory footprint of tokens already processed. They are complementary techniques.
- **Context Window Extension**: Compression is an alternative to architectural approaches (RoPE scaling, ALiBi) for fitting more information into limited context windows.
- **Tokenization**: Compression operates at the token level, so its effectiveness depends on the tokenizer -- subword tokenizers may split semantically important words into multiple tokens with different importance scores.
- **Knowledge Distillation**: LLMLingua-2's training process distills GPT-4's compression judgment into a small classifier, a form of task-specific knowledge distillation.
- **Model Routing**: Compression and routing are complementary cost-reduction strategies. Routing selects a cheaper model; compression reduces the input cost for whichever model is selected.

## Diagrams and Visualizations

*Recommended visual: LLMLingua pipeline showing budget controller, iterative token compression, and distribution alignment — see [LLMLingua Paper (arXiv:2310.05736)](https://arxiv.org/abs/2310.05736)*

*Recommended visual: Token-level perplexity scoring showing how low-perplexity (predictable) tokens are pruned first — see [LLMLingua-2 Paper (arXiv:2403.12968)](https://arxiv.org/abs/2403.12968)*

## Further Reading

- Jiang, H., Wu, Q., Luo, X., Li, D., Lin, C.-Y., Yang, Y., & Qiu, L. (2023). "LLMLingua: Compressing Prompts for Accelerated Inference of Large Language Models." EMNLP 2023. arXiv:2310.05736.
- Jiang, H., Wu, Q., Lin, C.-Y., Yang, Y., & Qiu, L. (2023). "LongLLMLingua: Accelerating and Enhancing LLMs in Long Context Scenarios via Prompt Compression." arXiv:2310.06839.
- Pan, Z., Wu, Q., Jiang, H., Xia, M., Luo, X., Zhang, J., Lin, C.-Y., Qiu, L., & Yang, Y. (2024). "LLMLingua-2: Data Distillation for Efficient and Faithful Task-Agnostic Prompt Compression." ACL 2024. arXiv:2403.12968.
- Li, Y., Bubeck, S., Eldan, R., Del Giorno, A., Gunasekar, S., & Lee, Y. T. (2023). "Selective Context: Efficient Inference of Large Language Models." arXiv:2310.06201.
- Wingate, D., Shoeybi, M., & Sorensen, T. (2022). "Prompt Compression and Contrastive Conditioning for Controllability and Toxicity Reduction in Language Models." EMNLP Findings 2022.
