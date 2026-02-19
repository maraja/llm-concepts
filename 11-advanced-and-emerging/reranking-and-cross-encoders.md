# Reranking and Cross-Encoders

**One-Line Summary**: Reranking is a second-stage retrieval technique where a more powerful model (typically a cross-encoder) re-scores and reorders the initial retrieval results from a fast first-stage retriever (bi-encoder or BM25), dramatically improving precision by jointly processing each query-document pair rather than comparing independent embeddings -- making two-stage "retrieve then rerank" the standard architecture for production retrieval systems.

**Prerequisites**: Understanding of bi-encoder embedding models (encode query and document separately into single vectors), vector similarity search for retrieval, the speed-accuracy trade-off in neural retrieval, and the basic RAG pipeline (retrieve top-k documents, then generate).

## What Is Reranking?

First-stage retrieval is designed for speed. Whether using BM25 (keyword matching) or bi-encoders (dense embedding similarity), the goal is to quickly narrow millions of candidate documents down to a manageable set (typically 50-1000 candidates). Speed is prioritized over precision because every document in the corpus must be considered.

*Recommended visual: Two-stage retrieve-then-rerank architecture: fast bi-encoder retrieves candidates, cross-encoder reranks top-k — see [Hugging Face Cross-Encoders Documentation](https://www.sbert.net/docs/cross_encoder/usage/usage.html)*


But this speed comes at a cost. Bi-encoders compress entire documents into single vectors, losing fine-grained information. BM25 matches keywords without understanding semantics. Both produce noisy rankings where the top-10 results often include irrelevant documents and miss relevant ones that are ranked lower.

Reranking adds a second stage: take the initial candidate set from the first-stage retriever, and re-score each candidate using a more powerful (but slower) model that can assess relevance with much higher accuracy. The reranker reorders the candidates so that the most relevant documents are at the top, and the least relevant are at the bottom or filtered out entirely.

The dominant reranking architecture is the **cross-encoder**, which processes the query and document *together* through a transformer, allowing full bidirectional attention between query and document tokens. This joint processing is far more accurate than the independent encoding used by bi-encoders, but it is also far slower -- which is why it is used only on the small candidate set from the first stage, not on the full corpus.

## How It Works


*Recommended visual: Bi-encoder vs cross-encoder architecture showing independent vs joint query-document processing — see [Sentence-BERT Paper (arXiv:1908.10084)](https://arxiv.org/abs/1908.10084)*

### Bi-Encoder vs. Cross-Encoder Architecture

Understanding the architectural difference between bi-encoders and cross-encoders is essential to understanding why reranking works:

**Bi-Encoder (First-Stage Retriever)**

```
Query: "effects of sleep deprivation"  -->  [Encoder]  -->  query_vector (768-dim)
Document: "Studies show that lack of..."  -->  [Encoder]  -->  doc_vector (768-dim)
Score = cosine_similarity(query_vector, doc_vector)
```

- Query and document are encoded *independently*.
- Document vectors can be pre-computed and indexed offline.
- Retrieval is a fast nearest-neighbor search (milliseconds over millions of documents).
- **Weakness**: The encoder must decide what information to keep in a single vector *without knowing what the query will be*. This information bottleneck limits accuracy.

**Cross-Encoder (Reranker)**

```
Input: "[CLS] effects of sleep deprivation [SEP] Studies show that lack of... [SEP]"
         |
         v
    [Full Transformer: all query tokens attend to all document tokens]
         |
         v
    [CLS] token embedding --> Linear layer --> Relevance score (scalar)
```

- Query and document are concatenated and processed *together* through the transformer.
- Every query token attends to every document token (and vice versa) through self-attention.
- The model can identify precisely how each query term relates to each document term.
- **Weakness**: Cannot pre-compute document representations. Must run the full transformer for every (query, document) pair. Processing 100 candidates takes ~100x the compute of processing one.

### The Two-Stage Pipeline

```
User Query
    |
    v
[Stage 1: First-Stage Retrieval]  -- Fast, low precision
    |  BM25 or Bi-Encoder
    |  Search full corpus (millions of docs)
    |  Return top-100 candidates (milliseconds)
    v
[Stage 2: Reranking]  -- Slow, high precision
    |  Cross-Encoder scores each (query, candidate) pair
    |  Reorder the 100 candidates by cross-encoder score
    |  Return top-5 or top-10 to the user/LLM
    v
[Generation or Display]
```

This two-stage architecture captures the best of both worlds:
- **Stage 1** provides high recall (the relevant documents are somewhere in the top-100) at low latency.
- **Stage 2** provides high precision (the relevant documents are at the top of the reranked list) with acceptable latency (because only 100 candidates need scoring, not millions).

### How Cross-Encoders Are Trained

Cross-encoder rerankers are typically trained on relevance-labeled data:

**Training data**: Pairs of (query, document) with binary relevance labels (relevant/not relevant) or graded relevance scores (0-3). Large-scale datasets include MS MARCO (~500K queries with relevant passages), Natural Questions, and domain-specific datasets.

**Loss function**: Binary cross-entropy for binary relevance, or margin-based losses that ensure relevant documents score higher than irrelevant ones. Some models use knowledge distillation from larger teacher models.

**Base model**: Most cross-encoder rerankers are initialized from pretrained language models (BERT, RoBERTa, DeBERTa, or specialized models). The [CLS] token representation is passed through a linear layer to produce a scalar relevance score.

**Hard negative mining**: Training effectiveness depends heavily on the quality of negative examples. Hard negatives -- documents that are retrieved by the first-stage system but are actually irrelevant -- are far more informative than random negatives. Training on hard negatives teaches the cross-encoder to distinguish between "looks relevant" and "is relevant."

### Prominent Reranking Models

**Cohere Rerank**

Cohere offers a commercial reranking API (Rerank v3, v3.5) that accepts a query and a list of documents, returning relevance scores:
- Closed-source cross-encoder hosted as an API
- Supports multilingual reranking (100+ languages)
- Handles documents up to 4096 tokens
- Typical latency: 100-500ms for reranking 100 documents
- Strong performance on BEIR and MTEB benchmarks
- Simple API: one call with query + documents, returns ranked results with scores
- Widely integrated into RAG frameworks (LangChain, LlamaIndex, Haystack)

**BGE Reranker (BAAI)**

The Beijing Academy of Artificial Intelligence (BAAI) released open-source reranking models as part of the BGE (BAAI General Embedding) family:
- **bge-reranker-base**: 278M parameters, based on XLM-RoBERTa
- **bge-reranker-large**: 560M parameters, stronger but slower
- **bge-reranker-v2-m3**: Multilingual, supports 100+ languages
- Fully open-source (MIT license), can be self-hosted
- Competitive with commercial alternatives on BEIR benchmarks
- Available via Hugging Face and sentence-transformers library

**RankGPT / LLM-Based Rerankers**

Sun et al. (2023) introduced RankGPT, which uses GPT-4 or other large language models as rerankers through prompting:
- The LLM is given the query and a list of candidate passages and asked to rank them by relevance
- Uses a sliding window approach: rank subsets of candidates, then merge the ranked sublists
- No training required -- uses the LLM's zero-shot understanding of relevance
- Surprisingly effective: GPT-4 as a reranker matches or outperforms purpose-trained cross-encoders on several benchmarks
- **Much slower and more expensive** than purpose-trained cross-encoders (seconds per query vs. milliseconds)
- Best suited for high-value, low-volume applications where accuracy justifies the cost

**Jina Reranker**

Jina AI offers both open-source and API-based rerankers:
- jina-reranker-v2-base-multilingual: Open-source, 278M parameters
- Strong multilingual support
- Available via API and Hugging Face

**Mixedbread Reranker**

Mixedbread.ai released competitive open-source rerankers:
- mxbai-rerank-large-v1: Strong performance on MTEB leaderboard
- Based on DeBERTa architecture

### ColBERT as Reranker

ColBERT (late interaction) can serve as either a first-stage retriever or a reranker. When used as a reranker, it provides a middle ground between bi-encoder speed and cross-encoder accuracy. See the dedicated ColBERT concept document for details.

## Benchmark Performance

On the BEIR benchmark (a diverse collection of 18 retrieval datasets across different domains):

| Model Type | Typical NDCG@10 | Latency (100 docs) |
|-----------|-----------------|---------------------|
| BM25 (keyword) | 0.40-0.45 | <10ms |
| Bi-encoder (e.g., BGE-large) | 0.50-0.55 | <50ms |
| Cross-encoder reranker (on top of bi-encoder) | 0.55-0.62 | 200-500ms |
| ColBERT reranker | 0.53-0.58 | 50-200ms |
| GPT-4 as reranker (RankGPT) | 0.56-0.63 | 2-10s |

The cross-encoder reranker typically improves NDCG@10 by **5-15 percentage points** over the first-stage retriever alone. This is one of the highest-impact, lowest-complexity improvements available in a retrieval pipeline.

On the MTEB (Massive Text Embedding Benchmark) reranking leaderboard, as of early 2025:
- Top cross-encoder models achieve scores of 60-68 on the reranking subset
- The gap between top and bottom models is substantial, indicating that reranker quality matters

## Why It Matters

**Two-stage retrieval is standard practice in production RAG systems.** The performance improvement from reranking is so consistent and significant that it is rare to see a production retrieval system that does not include a reranking stage. The reasons:

1. **Large, consistent quality improvement**: 5-15 point NDCG improvement with minimal engineering effort. Adding a reranker is typically the single highest-impact optimization for a RAG system.

2. **Architecture simplicity**: The first-stage retriever and the reranker can be developed, optimized, and swapped independently. Upgrading the reranker does not require re-indexing the corpus.

3. **Cost-effectiveness**: Cross-encoder reranking of 100 documents takes 200-500ms and costs fractions of a cent (for self-hosted models). This is a tiny cost relative to the LLM generation step, with a large quality payoff.

4. **Handles vocabulary mismatch**: Cross-encoders excel at recognizing relevance even when the query and document use completely different terminology. The joint attention mechanism can learn that "cardiac arrest" and "heart attack" are related, even if the bi-encoder embeddings place them far apart.

5. **Precision matters for RAG**: In RAG, you typically feed the top-3 to top-10 retrieved documents to the LLM. If any of those documents are irrelevant, they can confuse the LLM and degrade generation quality. Reranking ensures the top positions contain the most relevant documents, directly improving RAG answer quality.

6. **Reduces the "lost in the middle" problem**: Research shows LLMs pay more attention to documents at the beginning and end of the context window. By reranking so the most relevant document is first, you ensure the LLM focuses on the best evidence.

## Key Technical Details

- **First-stage candidate count**: The number of candidates to retrieve for reranking (typically called `top_n` or `fetch_k`) is a critical parameter. Too few (e.g., 10) risks missing relevant documents that the first-stage retriever ranked low. Too many (e.g., 1000) increases reranking latency. Typical values: 50-200 candidates for reranking, with the final top-k (5-10) passed to generation.

- **Reranker input length**: Cross-encoders have maximum input length limits (typically 512 tokens). Documents longer than this must be truncated or split. For long documents, a common approach is to rerank individual chunks and then aggregate scores per document.

- **Score calibration**: Cross-encoder scores are not calibrated probabilities. A score of 0.8 from one model does not mean the same thing as 0.8 from another model, or even 0.8 on a different query from the same model. Scores are useful for ranking within a single query's results, not for absolute relevance judgments. For filtering (removing irrelevant documents), use a threshold tuned on validation data.

- **Hybrid retrieval + reranking**: The most robust first-stage retrieval combines BM25 (keyword) and dense embedding (semantic) results using reciprocal rank fusion (RRF). The merged list is then reranked by the cross-encoder. This three-component pipeline (BM25 + bi-encoder + cross-encoder) is the gold standard for production retrieval.

- **Batch processing**: Cross-encoder inference is embarrassingly parallel. All (query, document) pairs can be processed in a single batched forward pass on a GPU, making reranking efficient even for large candidate sets.

- **Distilled rerankers**: To reduce latency, large cross-encoder models can be distilled into smaller, faster models with minimal accuracy loss. A 6-layer distilled model may achieve 90% of a 12-layer model's accuracy at 2x the speed.

- **Late interaction as middle ground**: ColBERT provides a middle ground between bi-encoders and cross-encoders. It can be used as a reranker (faster than cross-encoders, more accurate than bi-encoders) or as a first-stage retriever (slower than bi-encoders, more accurate).

## Common Misconceptions

**"A better embedding model eliminates the need for reranking."** Even the best bi-encoder embedding models benefit from cross-encoder reranking. The architectural limitation of bi-encoders (independent encoding, single-vector compression) means they will always miss some relevance signals that cross-encoders capture. The gap narrows with better embeddings but does not close.

**"Reranking is too slow for production."** Cross-encoder reranking of 100 documents takes 200-500ms on a single GPU. For most RAG applications, total latency is dominated by the LLM generation step (1-5 seconds), so the reranking overhead is minor. For extremely latency-sensitive applications, ColBERT-style rerankers or distilled models can reduce reranking to under 100ms.

**"GPT-4 is the best reranker."** GPT-4 is surprisingly effective at reranking (RankGPT), but purpose-trained cross-encoders are 100-1000x faster and often achieve comparable accuracy. LLM-based reranking is best reserved for offline evaluation or very high-value, low-volume scenarios.

**"Cross-encoders can replace bi-encoders entirely."** Cross-encoders cannot perform first-stage retrieval because they require processing every (query, document) pair. With a million documents, this means a million transformer forward passes per query. The two-stage architecture exists precisely because cross-encoders are too slow for full-corpus retrieval.

**"Reranking just reorders the same results."** Reranking can dramatically change the ranking order. A document at position 50 in the first-stage retrieval might move to position 1 after reranking, and a document at position 1 might drop to position 30. The reranker's cross-attention reveals relevance signals that the first-stage retriever missed entirely.

**"All cross-encoder rerankers are the same."** Reranker quality varies significantly based on the base model, training data, and training methodology. On the MTEB reranking leaderboard, there is a 10+ point gap between the best and worst cross-encoder models. Model choice matters.

## Connections to Other Concepts

- **RAG**: Reranking is a critical component of production RAG pipelines, improving the precision of retrieved documents before they are passed to the LLM for generation.
- **Embedding models and vector databases**: Bi-encoders (embedding models) handle first-stage retrieval; cross-encoders (rerankers) handle second-stage refinement. They are complementary, not competing.
- **ColBERT and late interaction**: ColBERT provides a middle-ground architecture between bi-encoders and cross-encoders, usable as either a retriever or a reranker.
- **Corrective RAG (CRAG)**: CRAG's relevance evaluation step can use a cross-encoder reranker as the relevance scorer, applying a threshold to determine whether retrieved documents are "correct," "incorrect," or "ambiguous."
- **Agentic RAG**: Agentic systems can dynamically decide whether to apply reranking based on first-stage retrieval confidence, saving latency when retrieval is already high-quality.
- **HyDE**: HyDE improves first-stage recall (the relevant documents appear somewhere in the candidate set). Reranking improves precision (the relevant documents are at the top). They are complementary: HyDE + reranking is a powerful combination.

## Further Reading

- Nogueira, R. & Cho, K. (2019). "Passage Re-ranking with BERT." (arXiv: 1901.04085) One of the earliest papers demonstrating BERT-based cross-encoder reranking, establishing the modern reranking paradigm.
- Sun, W. et al. (2023). "Is ChatGPT Good at Search? Investigating Large Language Models as Re-Ranking Agents." (arXiv: 2304.09542) The RankGPT paper showing that LLMs can perform competitive reranking through prompting.
- Xiao, S. et al. (2024). "BGE M3-Embedding: Multi-Lingual, Multi-Functionality, Multi-Granularity Text Embeddings Through Self-Knowledge Distillation." (arXiv: 2402.03216) Describes the BGE embedding and reranker family.
- Thakur, N. et al. (2021). "BEIR: A Heterogeneous Benchmark for Zero-shot Evaluation of Information Retrieval Models." *NeurIPS 2021.* The primary benchmark for evaluating retrieval and reranking models across diverse domains.
- Khattab, O. & Zaharia, M. (2020). "ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction over BERT." *SIGIR 2020.* The ColBERT paper, which provides an alternative reranking approach via late interaction.
