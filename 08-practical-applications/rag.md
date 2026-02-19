# Retrieval-Augmented Generation (RAG)

**One-Line Summary**: RAG grounds LLM responses in external knowledge by retrieving relevant documents at query time and injecting them into the prompt, dramatically reducing hallucination and enabling models to answer questions about data they were never trained on.

**Prerequisites**: Understanding of how LLMs generate text, the concept of embeddings (text as vectors), similarity search basics, and awareness of the context window and its limitations.

## What Is RAG?

Imagine an open-book exam. Instead of relying purely on memory (which may be outdated or contain gaps), you get to look up relevant passages in a textbook before answering each question. RAG gives LLMs the same advantage.

*Recommended visual: RAG pipeline showing document chunking, embedding, vector storage, retrieval, and augmented generation — see [Pinecone RAG Guide](https://www.pinecone.io/learn/retrieval-augmented-generation/)*


LLMs are trained on static datasets with knowledge cutoff dates. They cannot know about events after training, proprietary company documents, or rapidly changing information. RAG solves this by adding a retrieval step: before generating a response, the system searches a knowledge base for relevant information, then feeds that information into the prompt alongside the user's question. The model generates its answer grounded in the retrieved evidence.

The term was coined by Lewis et al. (2020), but the pattern -- retrieve then generate -- has become the dominant architecture for production LLM applications. If fine-tuning teaches a model new knowledge permanently, RAG gives it temporary, task-specific knowledge at inference time.

## How It Works

The RAG pipeline has three major phases: **indexing**, **retrieval**, and **generation**.

*Recommended visual: Comparison of parametric knowledge (in model weights) vs non-parametric knowledge (retrieved documents) — see [RAG Paper (arXiv:2005.11401)](https://arxiv.org/abs/2005.11401)*


### Phase 1: Indexing (Offline)

Before any queries arrive, you prepare your knowledge base.

*Recommended visual: Complete RAG pipeline: document ingestion → chunking → embedding → vector store → retrieval → augmented generation — see [Pinecone RAG Guide](https://www.pinecone.io/learn/retrieval-augmented-generation/)*


**Chunking**: Raw documents (PDFs, web pages, databases) are split into smaller pieces called chunks. This is necessary because embedding models have limited context windows and because smaller chunks enable more precise retrieval. Strategies range from fixed-size splitting (every 512 tokens) to semantic chunking (splitting at topic boundaries).

**Embedding**: Each chunk is passed through an embedding model (like OpenAI's text-embedding-3-large, BGE, or Cohere Embed) that converts it into a dense vector -- a list of numbers that encodes the chunk's semantic meaning. Similar meanings produce similar vectors.

**Storing**: These vectors, along with metadata (source document, page number, timestamps), are stored in a vector database (Pinecone, Weaviate, Qdrant, Chroma, or pgvector). The database builds an index -- a data structure optimized for fast nearest-neighbor search.

### Phase 2: Retrieval (Online)

When a user asks a question:

**Query encoding**: The user's query is embedded using the same embedding model, producing a query vector in the same semantic space as the document chunks.

**Similarity search**: The vector database finds the k most similar chunks by comparing the query vector against all stored vectors, using metrics like cosine similarity. Approximate nearest neighbor (ANN) algorithms like HNSW make this fast even with millions of vectors.

**Reranking**: The initial retrieval results are optionally passed through a cross-encoder reranker (like Cohere Rerank or a ColBERT model) that scores each chunk's relevance to the query more accurately than embedding similarity alone. This is slower but significantly more precise, and it reorders the top-k results so the most relevant chunks appear first.

### Phase 3: Generation

The retrieved chunks are injected into the LLM's prompt, typically in a structured format:

```
Given the following context:
[Chunk 1]
[Chunk 2]
[Chunk 3]

Answer the user's question: {query}

If the context does not contain enough information, say so.
```

The model generates its response grounded in the provided context. Advanced implementations also ask the model to cite specific chunks, enabling verifiable, traceable answers.

## Naive RAG vs. Advanced RAG Patterns

**Naive RAG** follows the basic pipeline: embed query, retrieve top-k chunks, stuff them into a prompt, generate. This works surprisingly well for simple questions but breaks down on complex queries.

**Advanced RAG** introduces multiple refinements:

- **Query decomposition**: Breaking a complex question into sub-questions, retrieving for each, then synthesizing. "Compare the revenue models of Uber and Lyft" becomes two separate retrieval queries.
- **Hybrid search**: Combining dense vector search (semantic) with sparse keyword search (BM25). BM25 excels at exact term matching ("error code 0x8007045D"), while semantic search handles paraphrases and conceptual queries. Merging both with reciprocal rank fusion gives the best of both worlds.
- **Recursive retrieval**: First retrieving at the summary level (what document is relevant?), then drilling into specific chunks within that document.
- **HyDE (Hypothetical Document Embeddings)**: Asking the LLM to generate a hypothetical answer, then using that answer as the retrieval query. This often retrieves better chunks than the original question.
- **GraphRAG**: Building a knowledge graph from the document corpus, then using graph traversal to retrieve not just individual chunks but connected concepts. Microsoft's GraphRAG uses LLMs to extract entities and relationships, builds a community hierarchy, and generates summaries at different levels of abstraction for global queries.

## Long-Context vs. RAG

Modern models with 128K-1M+ token context windows raise a natural question: why not just stuff all documents into the prompt and skip retrieval entirely?

The trade-offs are nuanced. Long-context approaches are simpler (no chunking, no embeddings, no vector database) and avoid retrieval errors. However, they are expensive (cost scales linearly with tokens), slow (latency increases with context length), and studies show models struggle with information "lost in the middle" of very long contexts. RAG remains more cost-effective and precise for large knowledge bases, while long-context windows are excellent for smaller document sets where retrieval errors would be costly.

The emerging best practice is a hybrid: use RAG to narrow down to a manageable set of highly relevant documents, then leverage long context to include more of those documents than you could in a 4K-token window.

## Why It Matters

RAG is arguably the most important architectural pattern in production LLM applications. It solves the three biggest limitations of standalone LLMs: knowledge staleness (training cutoff), lack of proprietary knowledge (company data), and hallucination (grounding in evidence). Nearly every enterprise LLM deployment -- from customer support chatbots to internal knowledge assistants to legal research tools -- uses some form of RAG.

## Key Technical Details

- **Chunk overlap**: Including 10-20% overlap between adjacent chunks preserves context that might be split at chunk boundaries.
- **Metadata filtering**: Pre-filtering by metadata (date range, document type, department) before vector search dramatically improves relevance for scoped queries.
- **Top-k selection**: Typical values range from k=3 to k=10. More chunks provide more context but increase noise and cost.
- **Embedding model choice matters**: The embedding model and the generative model do not need to come from the same provider. Specialized embedding models (like BGE or GTE) often outperform general-purpose models on retrieval benchmarks.
- **Evaluation**: RAG systems should be evaluated on both retrieval quality (recall, precision, MRR) and generation quality (faithfulness, relevance, completeness). Frameworks like RAGAS provide standardized metrics.

## Common Misconceptions

**"RAG eliminates hallucination."** RAG *reduces* hallucination by providing evidence, but the model can still hallucinate if the retrieved context is ambiguous, if it ignores the context, or if it extrapolates beyond what the context supports. Faithful generation requires additional engineering.

**"Better embeddings solve all retrieval problems."** Embeddings capture semantic similarity, but retrieval also fails due to poor chunking, missing metadata, ambiguous queries, and vocabulary mismatch. The full pipeline matters.

**"RAG is always better than fine-tuning."** They solve different problems. RAG adds dynamic, specific knowledge. Fine-tuning changes model behavior, style, or domain adaptation. Many production systems use both.

## Connections to Other Concepts

- **Embedding models and vector databases** are the infrastructure that makes RAG retrieval possible.
- **Chunking strategies** directly determine retrieval quality -- bad chunks mean bad retrieval, regardless of how good the embedding model is.
- **Prompt engineering** governs the generation phase -- how you structure the context-injection prompt affects answer quality.
- **Function calling** enables agentic RAG, where the model decides when to retrieve, what query to use, and whether to retrieve again.
- **AI agents** often use RAG as their memory and knowledge retrieval mechanism.

## Further Reading

- Lewis, P. et al. (2020). "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks." *NeurIPS 2020.* The foundational RAG paper.
- Gao, Y. et al. (2024). "Retrieval-Augmented Generation for Large Language Models: A Survey." A comprehensive survey of RAG techniques, from naive to advanced.
- Edge, D. et al. (2024). "From Local to Global: A Graph RAG Approach to Query-Focused Summarization." The Microsoft GraphRAG paper introducing community-based summarization.
