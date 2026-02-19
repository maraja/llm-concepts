# RAPTOR (Recursive Abstractive Processing for Tree-Organized Retrieval)

**One-Line Summary**: RAPTOR builds a hierarchical tree index over a document corpus by recursively clustering text chunks using UMAP and Gaussian mixture models, then summarizing each cluster with an LLM -- creating a multi-resolution representation where leaf nodes are original text chunks and higher nodes are increasingly abstract summaries, enabling retrieval at any level of detail from granular facts to high-level themes.

**Prerequisites**: Understanding of standard RAG chunking and retrieval, the limitation that flat chunk-based retrieval can only answer questions at the granularity of individual chunks, basic familiarity with clustering algorithms, dimensionality reduction, and the concept of hierarchical document representations.

## What Is RAPTOR?

Standard RAG systems split documents into fixed-size chunks and embed each chunk independently. This flat structure works well for questions that can be answered by a single chunk, but it fails for questions that require understanding across multiple chunks or at a higher level of abstraction.

*Recommended visual: RAPTOR tree structure showing leaf chunks clustered and summarized into hierarchical nodes — see [Sarthi et al. RAPTOR Paper (arXiv:2401.18059)](https://arxiv.org/abs/2401.18059)*


Consider a 50-page research paper. Standard RAG chunks it into ~100 pieces of 512 tokens each. If you ask "What specific results did the authors report in Table 3?", vector search will likely find the right chunk. But if you ask "What is the overall contribution of this paper?", no single chunk contains a complete answer. The abstract might help, but it does not capture the nuanced synthesis that comes from understanding the full paper.

RAPTOR, introduced by Sarthi, Abdullah, Tuli, Khanna, Goldie, and Manning at Stanford (2024), solves this by building a tree of summaries over the chunks. Adjacent or semantically related chunks are clustered together and summarized by an LLM. Those summaries are then clustered and summarized again, recursively, until you reach a root summary of the entire corpus (or major sections thereof). The tree preserves information at every level of granularity: leaf nodes contain specific details, intermediate nodes contain section-level summaries, and the root contains the highest-level themes.

At retrieval time, the system searches across all levels of the tree simultaneously, returning the most relevant nodes regardless of whether they are specific chunks or abstract summaries.

## How It Works


*Recommended visual: Multi-level retrieval in RAPTOR showing queries matching at different abstraction levels in the tree — see [RAPTOR Paper Figure 1](https://arxiv.org/abs/2401.18059)*

### Building the RAPTOR Tree (Indexing)

**Step 1: Leaf Node Creation**

Documents are split into text chunks, just as in standard RAG. The paper uses chunks of approximately 100 tokens (relatively small, to ensure each chunk captures a focused unit of information). Each chunk becomes a leaf node in the tree and is embedded using a sentence embedding model (the paper uses SBERT -- Sentence-BERT).

**Step 2: Clustering with UMAP + Gaussian Mixture Models (GMM)**

The leaf node embeddings are clustered using a two-step process:

1. **UMAP dimensionality reduction**: The high-dimensional embeddings (e.g., 768-dimensional) are projected to a lower-dimensional space (typically 10 dimensions) using UMAP (Uniform Manifold Approximation and Projection). UMAP preserves local and global structure better than alternatives like t-SNE and is computationally efficient. The reduced dimensionality makes clustering more reliable, avoiding the "curse of dimensionality" where distances become less meaningful in very high-dimensional spaces.

2. **Gaussian Mixture Model (GMM) clustering**: A soft-clustering GMM is fitted to the reduced-dimensional embeddings. Unlike hard clustering (like k-means, where each point belongs to exactly one cluster), GMM assigns each chunk a probability of belonging to each cluster. Chunks with high probability (above a threshold, typically 0.5) in a cluster are included in that cluster.

**Why soft clustering matters**: A chunk at the boundary between two topics -- say, a passage that discusses both methodology and results -- can belong to both the methodology cluster and the results cluster. This ensures that no information is lost at cluster boundaries, unlike hard clustering where each chunk is forced into a single group.

**Determining the number of clusters**: The optimal number of clusters is determined using the Bayesian Information Criterion (BIC). The algorithm tries different values of k and selects the one that best balances model fit with model complexity.

**Step 3: Summarization**

For each cluster, the text of all chunks assigned to that cluster is concatenated and sent to an LLM (the paper uses GPT-3.5 Turbo) with a prompt instructing it to generate a concise summary that captures the key information from all the included chunks. These summaries become the next level of nodes in the tree.

**Step 4: Recursive Application**

The summaries generated in Step 3 are themselves embedded and clustered using the same UMAP + GMM process. The resulting clusters are summarized again. This recursion continues until either:
- A single root summary is produced (for smaller documents)
- The remaining nodes are too few to meaningfully cluster further
- A maximum depth is reached

The result is a tree where:
- **Level 0 (leaves)**: Original text chunks (~100 tokens each)
- **Level 1**: Summaries of clusters of 3-10 related chunks (~200-500 tokens)
- **Level 2**: Summaries of summaries (section-level abstractions)
- **Level N (root)**: High-level summary of the entire document or corpus

Typical trees have 3-5 levels, depending on corpus size.

### Retrieval Strategies

RAPTOR supports two retrieval strategies:

**Tree Traversal (Top-Down)**

Start at the root and traverse down the tree:
1. Compare the query to the root-level summaries. Select the most relevant branch.
2. Compare the query to the children of the selected node. Select the most relevant child.
3. Continue until reaching the leaf level.
4. Return the selected nodes at the appropriate level(s).

This is efficient (logarithmic in the number of leaves) but may miss relevant information in branches not traversed.

**Collapsed Tree (Flat Search)**

Flatten all nodes from all levels into a single layer and perform standard top-k vector retrieval across the entire flattened set:
1. Embed all nodes (leaves + all summary levels).
2. Embed the query.
3. Retrieve the top-k most similar nodes, regardless of their level in the tree.

A query asking for specific details will naturally retrieve leaf nodes. A query asking for broad themes will retrieve higher-level summary nodes. The retrieval naturally adapts to the query's abstraction level.

**The paper found that the collapsed tree approach outperformed tree traversal** on most benchmarks, because it allows the retrieval to opportunistically select nodes at whatever level best matches the query, without being constrained by a top-down path.

## Benchmark Results

RAPTOR was evaluated on three question-answering benchmarks that test different levels of comprehension:

**QuALITY (Question Answering with Long Input Texts, Yes!)**
- Tests understanding of long narratives (stories, articles of 3,000-8,000 tokens)
- RAPTOR + GPT-4: **82.6%** accuracy
- Standard RAG (SBERT retrieval + GPT-4): **78.2%** accuracy
- Improvement: +4.4 percentage points

**QASPER (Question Answering on Scientific Papers)**
- Tests understanding of NLP research papers
- RAPTOR + GPT-4 (collapsed tree): **36.6** F1
- Standard RAG (DPR retrieval): **32.0** F1
- UnifiedQA (no retrieval): **27.4** F1

**NarrativeQA**
- Tests comprehension of full books and movie scripts
- RAPTOR + GPT-4: achieved state-of-the-art results, significantly outperforming flat retrieval baselines

The improvements were most pronounced on questions requiring higher-level understanding -- questions about themes, overall arguments, or connections between sections. For highly specific factual questions, RAPTOR performed comparably to standard RAG (since both retrieve the same leaf-level chunks).

## Why It Matters

RAPTOR addresses a fundamental limitation of flat-chunk RAG: the loss of hierarchical structure in documents. Human understanding of documents is naturally hierarchical -- we understand the overall argument, the section-level points, and the specific details. Standard RAG only captures the specific details level.

**Multi-resolution retrieval**: RAPTOR enables a single retrieval system to answer questions at any level of abstraction. "What was the GDP growth rate in Q3?" retrieves a specific chunk. "What are the key economic trends this year?" retrieves a higher-level summary. No other retrieval architecture provides this flexibility from a single index.

**Better long-document QA**: For documents longer than the LLM's context window, RAPTOR provides a principled way to compress and retrieve information. The tree structure means that even if you can only retrieve 5 nodes, those 5 nodes can span the entire document's content through summary nodes.

**Complementary to other techniques**: RAPTOR's tree structure can be combined with other RAG enhancements -- reranking, HyDE, agentic retrieval -- because it is an indexing-time technique that improves the quality of what is available to retrieve.

## Key Technical Details

- **UMAP parameters**: The paper uses `n_neighbors=10`, `min_dist=0.0`, `metric='cosine'`, `n_components=10` as default UMAP settings. The low `min_dist` encourages tight clusters; 10 components preserve enough structure for GMM clustering.
- **GMM threshold**: A chunk is assigned to a cluster if its probability of belonging to that cluster exceeds 0.5. With soft clustering, chunks can appear in multiple clusters, so some information is duplicated across summaries. This redundancy is intentional -- it ensures completeness.
- **Summary length**: Cluster summaries are typically 100-500 tokens, depending on the number and length of chunks in the cluster. The LLM is instructed to produce concise but comprehensive summaries.
- **Embedding model**: The paper uses SBERT (all-mpnet-base-v2) for embedding chunks and summaries. Any embedding model can be used; the choice affects clustering quality and retrieval accuracy.
- **Summarization model**: GPT-3.5 Turbo was used for summarization. Using a stronger model (GPT-4) improves summary quality at higher cost. Open-source models can also be used.
- **Token efficiency**: Because summaries compress multiple chunks into fewer tokens, the RAPTOR tree enables the LLM to "see" more of the document's content within a fixed retrieval budget. Retrieving 5 summary nodes might cover information from 50 original chunks.
- **Indexing overhead**: Building the tree requires O(N) embedding computations, O(N) UMAP+GMM operations, and O(N/k) LLM summarization calls per level, where N is the number of chunks and k is the average cluster size. Total LLM calls scale as approximately N/k * log_k(N). For a 100-page document with 200 chunks, expect roughly 30-60 LLM summarization calls.
- **Incremental updates**: Adding new documents requires recomputing the tree (or at minimum, the affected clusters). This is a limitation for rapidly changing corpora, though the affected clusters can be localized using tree structure.

## Common Misconceptions

**"RAPTOR is the same as recursive summarization."** Simple recursive summarization creates a linear chain of summaries (summarize chapters, then summarize the chapter summaries). RAPTOR uses clustering to group semantically related chunks regardless of their position in the document. A cluster might include chunks from Chapter 2 and Chapter 7 that discuss the same topic. This semantic clustering produces more coherent and useful summaries than position-based approaches.

**"RAPTOR only works for single long documents."** While the examples often focus on single documents, RAPTOR works across multiple documents. The clustering step groups semantically similar chunks regardless of which document they came from, naturally creating cross-document thematic summaries.

**"Higher tree levels are always better for abstract queries."** The collapsed tree approach works precisely because the right level depends on the specific query. Sometimes a mid-level summary is most relevant; sometimes a specific chunk is needed even for a seemingly abstract question. Letting the retrieval system choose freely across all levels outperforms forcing traversal from the top.

**"RAPTOR replaces the need for good chunking."** RAPTOR still depends on the quality of the leaf-level chunks. Poor chunking (splitting mid-sentence, chunks too large or too small) propagates errors up the tree. RAPTOR enhances good chunking but does not fix bad chunking.

**"The summaries lose important details."** By construction, the original leaf chunks are always in the tree. Summaries add higher-level representations but do not replace the originals. The collapsed tree retrieval can return leaf nodes when specificity is needed.

## Connections to Other Concepts

- **RAG**: RAPTOR is an advanced indexing strategy for RAG that replaces flat chunk storage with a hierarchical tree structure.
- **Chunking strategies**: RAPTOR depends on quality chunking for its leaf nodes. Semantic chunking at the leaf level can further improve RAPTOR's effectiveness.
- **GraphRAG**: Both RAPTOR and GraphRAG address the limitation of flat retrieval for global queries. GraphRAG uses entity-relationship graphs and community detection; RAPTOR uses clustering-based trees. They are complementary approaches to the same problem.
- **Embedding models**: RAPTOR uses embeddings for both clustering (via UMAP) and retrieval (via similarity search). The quality of the embedding model affects both stages.
- **Agentic RAG**: An agentic system could use RAPTOR's tree structure strategically -- first retrieving high-level summaries to understand the landscape, then drilling into specific leaf nodes for details.
- **Late chunking**: Both RAPTOR and late chunking aim to preserve broader context that is lost in standard chunking. Late chunking preserves context at the embedding level; RAPTOR preserves it through hierarchical summarization.

## Further Reading

- Sarthi, P., Abdullah, S., Tuli, A., Khanna, S., Goldie, A., & Manning, C. D. (2024). "RAPTOR: Recursive Abstractive Processing for Tree-Organized Retrieval." *ICLR 2024.* (arXiv: 2401.18059) The foundational RAPTOR paper from Stanford.
- McInnes, L., Healy, J., & Melville, J. (2018). "UMAP: Uniform Manifold Approximation and Projection for Dimension Reduction." (arXiv: 1802.03426) The dimensionality reduction algorithm used in RAPTOR's clustering pipeline.
- Gao, Y. et al. (2024). "Retrieval-Augmented Generation for Large Language Models: A Survey." Covers hierarchical retrieval approaches including RAPTOR in the broader RAG landscape.
- Wu, Z. et al. (2024). "Retrieval Head Mechanistically Explains Long-Context Factuality." Provides complementary insights into how LLMs use retrieved information from different context levels.
