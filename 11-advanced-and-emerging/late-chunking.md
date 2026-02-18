# Late Chunking

**One-Line Summary**: Late chunking reverses the traditional "chunk then embed" pipeline by first passing the entire document through the embedding model's transformer layers to produce contextualized token representations, then chunking those rich token embeddings into segment-level vectors -- preserving cross-chunk context that traditional chunking destroys.

**Prerequisites**: Understanding of how transformer encoder models produce token-level embeddings, the standard RAG indexing pipeline (chunk -> embed -> store), mean pooling for creating sentence/chunk embeddings, and awareness of how traditional chunking loses cross-boundary context.

## What Is Late Chunking?

In standard RAG pipelines, documents are chunked first and each chunk is embedded independently. This means the embedding model never sees the full document -- it processes each chunk in isolation. If a chunk says "He signed the contract on March 15th," the embedding model has no idea who "He" refers to because the antecedent appeared in a previous chunk. The coreference is lost, the embedding captures incomplete semantics, and retrieval quality suffers.

Late chunking, introduced by Gunther et al. (2024) at Jina AI, flips this pipeline. Instead of chunking first, you pass the entire document (or as much as fits in the model's context window) through the transformer encoder first. This produces a sequence of contextualized token embeddings where every token's representation is informed by every other token through self-attention. Then you apply chunk boundaries to these contextualized representations, mean-pooling the tokens within each chunk boundary to produce chunk-level embeddings.

The critical difference: in late chunking, each token embedding already encodes information about the full document context. When you chunk and mean-pool these rich representations, each chunk embedding implicitly knows about the rest of the document. The pronoun "He" in our example is now represented as a token embedding that already encodes the information about which person it refers to, because self-attention connected it to the antecedent during the full-document forward pass.

## How It Works

### The Standard Pipeline (Early Chunking)

```
Document -> Split into Chunks -> Embed each Chunk independently -> Store chunk embeddings
```

Each chunk is processed by the transformer in isolation. Token embeddings within chunk C_i only attend to other tokens in C_i. No cross-chunk information flows.

### The Late Chunking Pipeline

```
Document -> Embed full document (get token-level embeddings) -> Apply chunk boundaries -> Mean-pool tokens within each boundary -> Store chunk embeddings
```

Step by step:

**1. Full-document encoding**: The entire document is fed through the transformer encoder. For a document with N tokens, this produces N contextualized token embeddings, each of dimension d. Every token representation is informed by the full document through self-attention across all layers. This is the key: the transformer's attention mechanism serves as the cross-chunk context bridge.

**2. Determine chunk boundaries**: Chunk boundaries are determined using any standard chunking strategy (fixed-size, recursive, semantic, structure-aware). The boundaries are expressed as token index ranges: chunk_1 = tokens[0:128], chunk_2 = tokens[100:228] (with overlap), etc.

**3. Late pooling**: For each chunk boundary, the corresponding contextualized token embeddings are mean-pooled (averaged) to produce a single chunk-level embedding vector. This is identical to how standard embedding models pool token representations into sentence embeddings -- except the token representations are now informed by the full document.

**4. Storage and retrieval**: The resulting chunk embeddings are stored in a vector database exactly as in standard RAG. The retrieval process is unchanged. The improvement is entirely in the quality of the chunk embeddings.

### Why Self-Attention Is the Key Mechanism

The transformer's self-attention mechanism is what makes late chunking work. In a 12-layer transformer, each token's representation is iteratively refined by attending to all other tokens. By the final layer, each token embedding is a function of the entire input sequence, not just its local context. This means:

- Pronouns resolve to their referents: "He" near the end of a document becomes contextualized by the entity it refers to at the beginning.
- Domain-specific terms gain context: "Python" in a chunk about data analysis carries its programming-language meaning because the rest of the document is about software.
- Ambiguous phrases are disambiguated: "The bank" is contextualized as a financial institution or a river bank based on the full document.

When you mean-pool these rich token embeddings within chunk boundaries, the resulting chunk vector captures not just the chunk's local content but its contextual meaning within the broader document.

### Handling Long Documents

Most transformer encoders have context windows of 512-8192 tokens. Documents exceeding this limit cannot be processed in a single forward pass. Late chunking handles this through:

**Long-context embedding models**: Models like Jina Embeddings v2 (8192 tokens), NomicBERT (8192 tokens), and others with extended context windows can handle many documents in a single pass. The recent trend toward longer-context embedding models is partly motivated by enabling late chunking.

**Sliding window with overlap**: For documents exceeding even extended context windows, a sliding window approach processes overlapping segments. Tokens in overlapping regions get representations from multiple windows, which can be averaged or selected based on the window where they are most central.

**Hierarchical encoding**: Process sections of the document in separate forward passes, but include shared context (e.g., the document title and introduction) in each pass to provide some cross-section conditioning.

## Why It Matters

Late chunking addresses what is arguably the most fundamental weakness in standard RAG indexing: the loss of cross-chunk context. This loss manifests in several concrete failure modes:

- **Coreference resolution failure**: Chunks containing pronouns or references ("this approach," "the aforementioned study") produce poor embeddings because the referent is in another chunk.
- **Context-dependent terminology**: Technical terms, acronyms, and domain-specific jargon that are defined in one part of a document and used in another lose their grounding when chunks are embedded independently.
- **Narrative and argumentative structure**: In documents with logical flow (academic papers, legal contracts, technical reports), individual chunks often make partial sense without the broader argument they contribute to.

Late chunking mitigates all of these by ensuring every chunk embedding is informed by the full document context. In Jina AI's evaluations, late chunking showed consistent improvements on retrieval benchmarks -- particularly on datasets with long documents where cross-chunk dependencies are frequent. Improvements were most pronounced on tasks requiring understanding of document-level context, with gains of 5-15% on nDCG@10 compared to standard chunking on long-document retrieval benchmarks.

## Key Technical Details

- **Compute cost**: Late chunking requires one forward pass through the full document rather than separate passes through each chunk. For a document that produces K chunks, this is actually cheaper than standard chunking if K > 1, because you do one forward pass instead of K. However, the single pass processes a longer sequence, and self-attention is O(n^2) in sequence length, so the wall-clock time depends on the ratio of document length to chunk size and the model's context window.
- **Chunk boundary flexibility**: Because chunking happens after encoding, you can experiment with different chunking strategies on the same set of token embeddings without re-running the model. This enables rapid iteration on chunk sizes and boundaries.
- **Compatibility**: Late chunking works with any transformer encoder model that outputs token-level embeddings. It does not require model retraining -- you can apply it to existing embedding models, though models with longer context windows benefit more.
- **Overlap handling**: With late chunking, the tokens in overlapping regions between adjacent chunks share the same contextualized representations (since they came from the same forward pass). This means overlap adds storage cost but no additional compute.
- **Mean pooling vs. other strategies**: While mean pooling is the standard aggregation, weighted pooling (giving more weight to tokens central to the chunk) or attention-based pooling could further improve chunk embeddings. This is an active area of exploration.

## Common Misconceptions

**"Late chunking requires special embedding models."** It does not. Any encoder that produces token-level representations can be used. However, models with longer context windows allow larger documents to be processed in a single pass, maximizing the benefit. Models trained on short sequences (e.g., 512-token max) will have limited benefit for documents that exceed their context window.

**"Late chunking eliminates the need for chunk overlap."** Overlap is less critical with late chunking because cross-chunk context is preserved through the full-document attention. However, overlap can still be useful for ensuring key boundary information appears in multiple retrievable chunks.

**"Late chunking is the same as embedding the whole document."** It is not. Whole-document embedding produces a single vector for the entire document, which is too coarse for precise retrieval. Late chunking produces multiple vectors (one per chunk), each enriched with document-level context. It combines the precision of chunk-level retrieval with the context awareness of document-level encoding.

**"The benefits are marginal for short documents."** This is partially true. For documents that fit within a single chunk, there is no benefit. For documents that span only 2-3 chunks, the benefit exists but is smaller. The gains scale with document length and the frequency of cross-chunk dependencies.

## Connections to Other Concepts

- **Chunking strategies**: Late chunking is orthogonal to the choice of chunking strategy -- you still need to decide where to place chunk boundaries (fixed-size, semantic, structure-aware). Late chunking changes when the boundaries are applied, not how they are determined.
- **Embedding models and vector databases**: Late chunking produces the same output format (fixed-size vectors per chunk), so the vector database and retrieval pipeline are unchanged.
- **ColBERT / late interaction**: Both late chunking and ColBERT preserve token-level information that standard bi-encoder embedding discards. Late chunking preserves document context within chunks; ColBERT preserves per-token matching granularity.
- **Long-context models**: The trend toward longer context windows in both embedding models and generative models is synergistic with late chunking, enabling it to work on longer documents.
- **RAG**: Late chunking is an indexing-time improvement that transparently improves retrieval quality in any RAG pipeline without changing the query or generation phases.

## Further Reading

- Gunther, M. et al. (2024). "Late Chunking: Contextual Chunk Embeddings Using Long-Context Embedding Models." *Jina AI Technical Report.* (arXiv: 2409.04701) The paper introducing late chunking and demonstrating its benefits on retrieval benchmarks.
- Jina AI blog (2024). "Late Chunking in Long-Context Embedding Models." Practical guide to implementing late chunking with Jina Embeddings v2.
- Muennighoff, N. et al. (2023). "MTEB: Massive Text Embedding Benchmark." Provides the evaluation framework used to measure late chunking improvements.
