# Embedding Models & Vector Databases

**One-Line Summary**: Embedding models transform text into numerical vectors that capture semantic meaning, and vector databases store and search those vectors at scale, together forming the retrieval backbone of modern LLM applications.

**Prerequisites**: Basic understanding of what vectors are (lists of numbers representing points in space), familiarity with neural networks at a high level, and awareness of why LLM applications need to search over large document collections.

## What Is This?

Imagine a library where every book is placed on a shelf not alphabetically, but by *meaning*. Books about cooking are near each other, books about quantum physics cluster together, and a book about "the chemistry of baking" sits somewhere between the two groups. Embedding models are the librarians who decide where each book goes, and vector databases are the shelving systems designed to find the nearest books to any query instantly.

More precisely, an embedding model is a neural network that takes a piece of text (a sentence, paragraph, or document) and outputs a fixed-size vector -- a list of floating-point numbers, typically 384 to 3072 dimensions. These vectors are constructed so that texts with similar meanings have similar vectors, as measured by geometric distance or angle. A vector database is a specialized storage system optimized for storing millions or billions of such vectors and answering "find me the most similar vectors" queries in milliseconds.

## How It Works

### Text to Vectors: Embedding Models

Embedding models are typically transformer-based encoders (not decoders like GPT). They process input text through multiple layers of self-attention, then aggregate the output into a single vector. The aggregation strategy varies: some models use the [CLS] token embedding, others average all token embeddings (mean pooling), and some use more sophisticated approaches.

**Training** is what makes these vectors semantically meaningful. Modern embedding models are trained with contrastive learning: the model sees pairs of texts that should be similar (a question and its answer, a sentence and its paraphrase) and pairs that should be dissimilar, and it learns to push similar pairs closer together in vector space while pushing dissimilar pairs apart. This training signal teaches the model to encode meaning, not just surface-level word overlap.

**Popular embedding models** span a wide range:

- **OpenAI text-embedding-3-large** (3072 dimensions): Widely used commercial model with strong general-purpose performance and a Matryoshka representation that allows dimension truncation without retraining.
- **BGE (BAAI General Embedding)** family: Open-source models from the Beijing Academy of AI, competitive with commercial offerings. BGE-M3 supports multilingual, multi-granularity, and multi-functionality retrieval.
- **Cohere Embed v3**: Commercial model with strong multilingual support and compression-aware training.
- **GTE (General Text Embeddings)**: Alibaba's open-source models, strong on retrieval benchmarks.
- **E5 and E5-Mistral**: Microsoft's embedding models, with E5-Mistral being a decoder-based model that achieves strong performance by repurposing a generative LLM as an encoder.
- **Nomic Embed**: Open-source, open-data, fully auditable embedding model with competitive performance.

### Similarity Metrics

Once you have vectors, you need a way to measure how similar two vectors are:

**Cosine similarity** measures the angle between two vectors, ignoring magnitude. It ranges from -1 (opposite) to 1 (identical direction). This is the most commonly used metric because it is robust to differences in vector length and works well when the meaningful information is in the direction, not the magnitude, of vectors.

**Dot product** multiplies corresponding dimensions and sums the results. Unlike cosine similarity, it is sensitive to vector magnitude. When vectors are normalized (unit length), dot product equals cosine similarity. Some models are trained to produce normalized vectors, making the two metrics equivalent.

**Euclidean distance (L2)** measures the straight-line distance between two points in vector space. Smaller distances mean more similar vectors. It is sensitive to both direction and magnitude. Less commonly used for text retrieval but important in some applications.

The choice of metric should match the embedding model's training objective. Most modern models document which metric they were optimized for.

### Vector Databases

Storing a few hundred vectors in a list and doing brute-force comparison is trivial. Storing a hundred million vectors and finding the nearest neighbors in under 50 milliseconds requires specialized infrastructure.

**Pinecone**: Fully managed cloud service. No infrastructure to manage. Supports metadata filtering, namespaces, and sparse-dense hybrid search. Popular for teams that want simplicity.

**Weaviate**: Open-source with a managed cloud option. Supports hybrid search natively, has a GraphQL API, and offers built-in vectorization modules that can generate embeddings automatically.

**Qdrant**: Open-source, written in Rust for performance. Supports payload (metadata) filtering, quantization for memory efficiency, and has a strong focus on production reliability.

**Chroma**: Lightweight, open-source, designed for developer ergonomics. Excellent for prototyping and smaller-scale applications. Embeds directly into Python applications.

**pgvector**: A PostgreSQL extension that adds vector similarity search to an existing relational database. Ideal when you already use PostgreSQL and want to avoid introducing a new database system.

**FAISS (Facebook AI Similarity Search)**: A library, not a database. Provides highly optimized ANN algorithms. Often used as the search engine inside other systems or for offline batch processing.

### Approximate Nearest Neighbor (ANN) Algorithms

Exact nearest neighbor search requires comparing the query vector against every stored vector -- O(n) complexity that becomes impractical at scale. ANN algorithms trade a small amount of accuracy for massive speed improvements.

**HNSW (Hierarchical Navigable Small World)**: Builds a multi-layer graph where each node is a vector, connected to its approximate nearest neighbors. Search starts at the top layer (sparse, long-range connections) and descends to lower layers (dense, short-range connections), like zooming into a map. HNSW offers excellent query speed with high recall and is the default algorithm in most vector databases. Its trade-off is high memory usage.

**IVF (Inverted File Index)**: Partitions the vector space into clusters using k-means. At query time, only the closest clusters are searched, dramatically reducing the number of comparisons. IVF uses less memory than HNSW but typically has lower recall at the same query speed. Often combined with product quantization (IVF-PQ) to compress vectors and reduce memory further.

**Quantization** techniques (scalar, product, binary) compress vectors from 32-bit floats to smaller representations, reducing memory by 4-32x with modest accuracy loss. This is increasingly important as embedding dimensions grow.

## Why It Matters

Embeddings and vector databases are the infrastructure layer that enables RAG, semantic search, recommendation systems, anomaly detection, and deduplication. Without them, LLM applications would be limited to what fits in the context window or what the model memorized during training. They are the bridge between static model knowledge and dynamic, domain-specific information.

## Key Technical Details

- **Embedding dimensions**: Common sizes are 384 (lightweight models), 768 (BERT-scale), 1024 (many modern models), and 1536-3072 (large commercial models). Higher dimensions capture more nuance but require more storage and computation. Matryoshka embeddings allow truncation to lower dimensions with graceful degradation.
- **Batch embedding**: Embedding large corpora should be done in batches with rate limit management and progress checkpointing. Re-embedding is expensive.
- **Normalization**: If using cosine similarity, pre-normalizing vectors to unit length and then using dot product is computationally cheaper.
- **Index build time vs. query time**: HNSW indices are slow to build but fast to query. IVF indices are faster to build but slower to query per unit of recall.
- **Hybrid search**: Combining dense (embedding) and sparse (BM25/TF-IDF) retrieval with reciprocal rank fusion often outperforms either approach alone.

## Common Misconceptions

**"All embedding models are roughly the same."** Performance varies dramatically across models, tasks, and languages. The MTEB (Massive Text Embedding Benchmark) leaderboard shows significant gaps. Model choice matters.

**"More dimensions always means better quality."** Beyond a certain point, additional dimensions add noise rather than signal. A well-trained 768-dimensional model can outperform a poorly trained 3072-dimensional one.

**"Vector databases are just for RAG."** They are used for recommendation engines, image search (with CLIP embeddings), anomaly detection, near-duplicate detection, and any application requiring similarity matching.

**"You need a dedicated vector database."** For small-scale applications (under a million vectors), pgvector or even in-memory FAISS can be perfectly adequate. Dedicated vector databases become important at scale or when you need managed infrastructure.

## Connections to Other Concepts

- **RAG** depends entirely on embedding models and vector databases for its retrieval phase.
- **Chunking strategies** determine what text gets embedded -- the quality of chunks directly affects embedding quality.
- **Tokenization** matters because embedding models have their own tokenizers and context windows, separate from the generative LLM.
- **Attention mechanisms** are the building blocks inside embedding models themselves.
- **Fine-tuning** embedding models on domain-specific data (using techniques like contrastive fine-tuning) can dramatically improve retrieval quality for specialized domains.

## Diagrams and Visualizations

![Word embedding space showing how semantically similar words cluster together, with vector arithmetic like king - man + woman = queen](https://jalammar.github.io/images/word2vec/king-analogy-viz.png)
*Source: [Jay Alammar – The Illustrated Word2Vec](https://jalammar.github.io/illustrated-word2vec/)*

*Recommended visual: Vector database architecture showing embedding, indexing (HNSW/IVF), and similarity search pipeline — see [Pinecone – What is a Vector Database](https://www.pinecone.io/learn/vector-database/)*

## Further Reading

- Johnson, J., Douze, M., & Jegou, H. (2019). "Billion-scale similarity search with GPUs." *IEEE Transactions on Big Data.* The FAISS paper that established foundational ANN techniques.
- Muennighoff, N. et al. (2023). "MTEB: Massive Text Embedding Benchmark." *EACL 2023.* The standard benchmark for evaluating embedding models across diverse tasks.
- Malkov, Y. & Yashunin, D. (2020). "Efficient and Robust Approximate Nearest Neighbor Using Hierarchical Navigable Small World Graphs." *IEEE TPAMI.* The definitive HNSW paper.
