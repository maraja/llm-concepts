# GraphRAG (Graph-Based Retrieval-Augmented Generation)

**One-Line Summary**: GraphRAG augments standard RAG by constructing a knowledge graph of entities and relationships from the document corpus, applying hierarchical community detection, and generating community summaries at multiple levels of abstraction -- enabling both precise local retrieval and global sensemaking queries that standard vector-based RAG fundamentally cannot answer.

**Prerequisites**: Understanding of standard RAG pipelines (chunking, embedding, vector retrieval, generation), the limitations of vector similarity search for holistic or thematic queries, basic graph theory concepts (nodes, edges, communities), and awareness that LLMs can extract structured information from unstructured text.

## What Is GraphRAG?

Standard vector RAG excels at finding specific passages that are semantically similar to a query. Ask "What is the company's parental leave policy?" and vector search will find the relevant policy chunk. But ask "What are the main themes discussed across all board meeting transcripts this year?" and vector RAG fails completely. No single chunk contains the answer because the question requires synthesizing information scattered across the entire corpus. The query is *global* -- it asks about the dataset as a whole, not about a specific fact.

GraphRAG, introduced by Edge, Trinh, Cheng, et al. at Microsoft Research (2024), solves this by building a structured knowledge graph from the document corpus and then using that graph's hierarchical community structure to answer both local (specific) and global (thematic) queries.

The key insight is that a knowledge graph captures *relationships* between entities, not just the content of individual chunks. And by detecting communities of densely connected entities and pre-generating summaries of those communities, GraphRAG creates a multi-resolution map of the corpus that can answer questions at any level of abstraction.

The paper's title captures the aspiration precisely: "From Local to Global: A Graph RAG Approach to Query-Focused Summarization."

## How It Works

### The GraphRAG Indexing Pipeline (Offline)

GraphRAG's indexing pipeline is significantly more complex (and more expensive) than standard RAG indexing. It consists of several stages:

**Stage 1: Source Document Chunking**

Documents are split into text chunks, similar to standard RAG. The default chunk size in Microsoft's implementation is 300 tokens (configurable), with overlap. Each chunk is a unit of text that will be processed by the LLM for entity extraction.

**Stage 2: Entity and Relationship Extraction**

Each chunk is sent to an LLM (GPT-4 or similar) with a detailed prompt that instructs it to extract:
- **Entities**: Named people, organizations, locations, concepts, events, technologies, etc. Each entity has a name, type, and description.
- **Relationships**: Connections between entities, with a description of the relationship and a strength/weight score.

For example, from a chunk about climate policy, the LLM might extract:
- Entities: "Paris Agreement" (event), "UNFCCC" (organization), "carbon neutrality" (concept), "2050" (date)
- Relationships: "Paris Agreement" -> "carbon neutrality" (aims to achieve), "UNFCCC" -> "Paris Agreement" (administers)

This is done for every chunk in the corpus. The LLM typically makes multiple extraction passes over each chunk to catch entities and relationships that were missed on the first pass. The Microsoft implementation uses up to 3 "gleaning" passes by default.

**Stage 3: Graph Construction and Entity Resolution**

All extracted entities and relationships are merged into a single knowledge graph. Entity resolution merges duplicate entities that refer to the same real-world entity but may have different surface forms (e.g., "UN Framework Convention on Climate Change" and "UNFCCC"). The LLM is used for this disambiguation step as well, generating a consolidated description for each resolved entity.

The result is a graph where nodes are entities (with rich descriptions) and edges are relationships (with descriptions and weights).

**Stage 4: Hierarchical Community Detection (Leiden Algorithm)**

The graph is partitioned into communities using the Leiden algorithm, a state-of-the-art community detection method that identifies clusters of densely interconnected nodes. Crucially, Leiden produces a *hierarchical* community structure:

- **Level 0 (finest)**: Many small, tightly-knit communities of closely related entities.
- **Level 1**: Medium-sized communities formed by merging related Level 0 communities.
- **Level 2+ (coarsest)**: Large communities representing broad thematic areas of the corpus.

This hierarchy is the key data structure that enables multi-resolution queries. A question about a specific entity can be answered using Level 0 communities. A question about broad themes uses Level 2+ communities.

**Stage 5: Community Summary Generation**

For each community at each level of the hierarchy, an LLM generates a summary that describes:
- The key entities in the community
- The main relationships between them
- The overall theme or topic the community represents
- Key findings, claims, or insights associated with the community

These summaries are generated bottom-up: Level 0 summaries are generated from the raw entity/relationship descriptions. Higher-level summaries are generated from the summaries of their child communities.

These pre-computed summaries are the retrieval targets for global queries. They are stored alongside the graph structure and the original text chunks.

### Query Modes

GraphRAG supports two query modes:

**Global Search (Map-Reduce over Community Summaries)**

For global/thematic queries ("What are the main themes?", "Summarize the key findings across all documents"), GraphRAG uses a map-reduce approach:

1. **Map phase**: Each community summary at the selected hierarchy level is sent to the LLM with the query. The LLM generates a partial answer based on that community's content and assigns a relevance score (0-100).
2. **Reduce phase**: The partial answers are ranked by relevance, and the top partial answers are combined into a final comprehensive response.

This ensures every part of the corpus is considered (through its community summary) without needing to stuff the entire corpus into a single prompt. The community hierarchy allows controlling the granularity: using higher-level (coarser) communities reduces cost but sacrifices detail; using lower-level communities provides more detail at higher cost.

**Local Search (Entity-Centric Retrieval)**

For specific queries about particular entities or topics, GraphRAG:

1. Identifies the entities most relevant to the query (using entity description embeddings and vector similarity).
2. Retrieves the neighborhoods of those entities in the knowledge graph -- their direct relationships, connected entities, community memberships, and associated text chunks.
3. Assembles this entity-centric context and sends it to the LLM for generation.

Local search combines the precision of standard RAG (retrieving specific text chunks) with the relational context from the knowledge graph (understanding how entities connect).

### The Indexing Cost

GraphRAG's indexing pipeline is significantly more expensive than standard RAG because it requires many LLM calls:
- Entity/relationship extraction for every chunk (with multiple gleaning passes)
- Entity resolution/disambiguation
- Community summary generation at every hierarchy level

For a corpus of 1 million tokens, the Microsoft team reported indexing costs of approximately $2-6 using GPT-4 Turbo pricing (as of mid-2024). This is orders of magnitude more expensive than simply embedding chunks with a dense embedding model. However, this is a one-time offline cost, and the resulting graph structure enables query capabilities that are impossible with vector RAG alone.

## Comparison: GraphRAG vs. Standard Vector RAG

| Dimension | Standard Vector RAG | GraphRAG |
|-----------|-------------------|----------|
| **Query type** | Specific, factual questions | Both specific and global/thematic |
| **Retrieval unit** | Text chunks | Community summaries, entity neighborhoods, text chunks |
| **Global queries** | Fails (no single chunk contains holistic answers) | Excels (community summaries pre-aggregate information) |
| **Indexing cost** | Low (embedding only) | High (LLM-based extraction + community summarization) |
| **Indexing speed** | Minutes | Hours to days (depending on corpus size and LLM throughput) |
| **Storage** | Vector embeddings | Graph + community summaries + embeddings + raw chunks |
| **Relationships** | Implicit (in embedding space) | Explicit (graph edges with descriptions) |
| **Explainability** | Low (similarity scores) | High (entity paths, community membership, relationship descriptions) |
| **Latency** | Low (single vector search) | Higher for global (map-reduce), comparable for local |
| **Maintenance** | Re-embed new documents | Re-extract entities, update graph, regenerate community summaries |

## Benchmark Results

In the original paper, Edge et al. evaluated GraphRAG against baseline naive RAG on a corpus of podcast transcripts (approximately 1 million tokens) using LLM-as-judge evaluation with the following metrics:
- **Comprehensiveness**: How thorough is the answer?
- **Diversity**: How many different perspectives/aspects does the answer cover?
- **Empowerment**: How well does the answer help the user understand the topic?
- **Directness**: How directly does the answer address the query?

For global sensemaking queries, GraphRAG (using community summaries at intermediate hierarchy levels) won on comprehensiveness and diversity by substantial margins (often 70-80% win rate against naive RAG). Naive RAG performed comparably or slightly better on directness for specific factual queries, where its precision retrieval is sufficient.

The results clearly demonstrated that the two approaches are complementary: standard RAG for precise local queries, GraphRAG for global thematic queries.

## When to Use GraphRAG

**Use GraphRAG when:**
- Your queries require synthesizing information across the entire corpus or large portions of it
- You need thematic analysis, trend detection, or holistic summarization
- The corpus has rich entity-relationship structure (people, organizations, events, concepts with clear connections)
- Users ask "What are the main themes/patterns/trends?" questions
- Explainability matters -- you need to show entity connections and reasoning paths
- The corpus is relatively stable (not rapidly changing), justifying the indexing investment

**Use standard vector RAG when:**
- Queries are specific and factual ("What is X's policy on Y?")
- Low latency is critical
- The corpus changes frequently (high re-indexing cost with GraphRAG)
- Budget is limited (GraphRAG indexing is expensive)
- The document structure does not have strong entity-relationship patterns (e.g., creative writing, poetry)

**Use both when:**
- Production systems often route queries: simple factual queries go through standard RAG, while complex/thematic queries go through GraphRAG. This hybrid approach provides the best of both worlds.

## Why It Matters

GraphRAG represents a fundamental advance in what RAG systems can answer. Standard vector RAG is limited to questions that can be answered by one or a few text chunks -- it is inherently a "needle in a haystack" approach. GraphRAG enables "understanding the haystack" -- answering questions about the corpus as a whole.

For enterprise knowledge management, this is transformative. Analysts can ask "What are the main risk factors discussed across all quarterly reports?" instead of reading them all manually. Researchers can ask "What are the emerging themes in this collection of 500 papers?" Legal teams can ask "What entities are connected to this contract across all our agreements?"

The knowledge graph also provides a powerful foundation for explainability. Instead of a black-box retrieval score, GraphRAG can show which entities were identified, how they connect, and which community summaries contributed to the answer.

## Key Technical Details

- **Leiden algorithm**: The Leiden algorithm (Traag et al., 2019) is chosen for community detection because it produces high-quality hierarchical partitions efficiently and guarantees that communities are well-connected (no disconnected subgraphs within a community). It is an improvement over the earlier Louvain algorithm.
- **Gleaning passes**: The entity extraction prompt is run multiple times (default: up to 3 passes) on each chunk, with each subsequent pass asking the LLM to identify entities it missed previously. This improves recall at the cost of additional LLM calls.
- **Community summary token budget**: Each community summary is generated with a target length (e.g., 500-1000 tokens) to keep the map-reduce process manageable. Summaries at higher hierarchy levels are naturally longer because they cover more entities.
- **Entity description embeddings**: Entity nodes store both their textual descriptions and vector embeddings of those descriptions, enabling vector-based entity lookup for local search.
- **Open source**: Microsoft released GraphRAG as an open-source Python library (github.com/microsoft/graphrag) in mid-2024, with Apache 2.0 licensing.
- **LLM flexibility**: While the paper used GPT-4 Turbo, the indexing pipeline can use any capable LLM. Using cheaper models (GPT-3.5, open-source models) reduces indexing cost but may reduce entity extraction quality.
- **Scalability**: For very large corpora (tens of millions of tokens), the indexing pipeline can be parallelized across chunks. The community detection and summary generation steps scale with graph size rather than raw token count.

## Common Misconceptions

**"GraphRAG replaces standard RAG."** GraphRAG complements standard RAG. For precise factual queries, standard vector RAG is faster, cheaper, and equally effective. GraphRAG's value is specifically in enabling query types that vector RAG cannot handle. Most production systems should use both.

**"GraphRAG requires a pre-existing knowledge graph."** GraphRAG builds its knowledge graph from scratch using LLM-based extraction. You do not need an existing ontology, knowledge base, or curated graph. The LLM constructs the graph from raw text. However, if you have an existing knowledge graph, it can be incorporated to improve quality.

**"The LLM-extracted graph is perfect."** Entity extraction is noisy. The LLM will miss entities, extract spurious relationships, and sometimes hallucinate connections. The multiple gleaning passes and entity resolution steps mitigate this, but the graph will contain errors. For most applications, the graph does not need to be perfect -- it needs to be good enough to form useful community structures.

**"GraphRAG is too expensive for production."** The indexing cost is a one-time expense (per corpus update). The query-time cost depends on the query mode: local search is comparable to standard RAG, while global search (map-reduce) is more expensive. For corpora that change infrequently and where global queries provide high value, the cost is justified.

## Connections to Other Concepts

- **RAG**: GraphRAG extends standard RAG with graph-based retrieval and community summarization for global queries.
- **Chunking strategies**: GraphRAG still relies on text chunking as the first indexing step; chunk size affects entity extraction quality.
- **Embedding models and vector databases**: GraphRAG uses embeddings for entity description matching in local search mode, alongside the graph structure.
- **Agentic RAG**: An agentic system can dynamically choose between standard RAG and GraphRAG based on query complexity, routing simple queries through vector search and complex ones through graph-based retrieval.
- **Compound AI systems**: GraphRAG is a prime example of a compound AI system -- multiple LLM calls, graph algorithms, and retrieval mechanisms orchestrated into a pipeline.
- **Neurosymbolic AI**: GraphRAG combines neural capabilities (LLM extraction and generation) with symbolic structures (knowledge graphs), making it a form of neurosymbolic AI applied to retrieval.

## Diagrams and Visualizations

*Recommended visual: GraphRAG pipeline: document → entities/relationships → knowledge graph → community detection → community summaries — see [Microsoft GraphRAG Paper (arXiv:2404.16130)](https://arxiv.org/abs/2404.16130)*

*Recommended visual: Local vs global search in GraphRAG showing entity-level retrieval vs community-summary-level retrieval — see [Microsoft GraphRAG GitHub](https://github.com/microsoft/graphrag)*

## Further Reading

- Edge, D., Trinh, H., Cheng, N., et al. (2024). "From Local to Global: A Graph RAG Approach to Query-Focused Summarization." (arXiv: 2404.16130) The foundational GraphRAG paper from Microsoft Research.
- Microsoft GraphRAG open-source repository: github.com/microsoft/graphrag. The reference implementation with detailed documentation and configuration guides.
- Traag, V. A., Waltman, L., & van Eck, N. J. (2019). "From Louvain to Leiden: guaranteeing well-connected communities." *Scientific Reports, 9*(1), 5233. The Leiden community detection algorithm used in GraphRAG.
- Gao, Y. et al. (2024). "Retrieval-Augmented Generation for Large Language Models: A Survey." Places GraphRAG in the broader taxonomy of advanced RAG techniques.
