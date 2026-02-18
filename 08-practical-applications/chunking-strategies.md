# Chunking Strategies for RAG

**One-Line Summary**: Chunking is the process of splitting documents into smaller pieces for embedding and retrieval, and the choice of chunking strategy directly determines whether a RAG system retrieves useful context or useless fragments.

**Prerequisites**: Understanding of what RAG (Retrieval-Augmented Generation) is and why it needs document chunks, basic familiarity with embedding models and their context window limits, and awareness of vector similarity search.

## What Are Chunking Strategies?

Imagine you have a 300-page textbook and someone asks you a specific question. You would not hand them the entire book -- that is too much to process. You also would not rip out individual sentences -- most sentences are meaningless without surrounding context. Instead, you would find the most relevant section, perhaps a paragraph or two, that contains the answer and enough surrounding context to make sense.

Chunking is this same decision, automated at scale. Before documents can be embedded and stored in a vector database, they must be split into pieces (chunks) that are small enough to be embedded meaningfully and retrieved precisely, but large enough to contain coherent, self-contained information. This seemingly simple task has an outsized impact on RAG quality: the best embedding model and the best LLM in the world cannot compensate for chunks that split a key fact across two pieces or bury a relevant sentence in a wall of irrelevant text.

## How It Works

### Fixed-Size Chunking

The simplest approach: split text into chunks of a fixed number of tokens or characters, regardless of content.

**How it works**: Set a chunk size (e.g., 512 tokens) and a step size. Walk through the document, cutting a new chunk every 512 tokens.

**Advantages**: Dead simple to implement, predictable chunk sizes, works for any document type.

**Disadvantages**: Cuts through sentences, paragraphs, and ideas indiscriminately. A key fact like "The contract expires on March 15, 2025, with a 30-day renewal window" might be split between two chunks, making neither chunk independently useful.

Despite its crudeness, fixed-size chunking with overlap is a surprisingly strong baseline. Many production systems start here and only move to more sophisticated strategies when they identify specific failure cases.

### Recursive Character/Text Splitting

The default strategy in LangChain and many RAG frameworks. It tries to split on natural boundaries in a priority order.

**How it works**: Given a list of separators (e.g., `["\n\n", "\n", ". ", " "]`), the algorithm first tries to split on double newlines (paragraph boundaries). If any resulting piece is still too large, it falls back to single newlines (line boundaries), then sentences, then words. This recursion continues until all chunks are below the target size.

**Advantages**: Respects document structure at multiple levels. Paragraphs stay together when possible. Much better than pure fixed-size for prose documents.

**Disadvantages**: Still purely syntactic -- it does not understand whether a paragraph is a self-contained thought or part of a multi-paragraph argument. The separator hierarchy is hardcoded and may not match all document formats.

### Semantic Chunking

Splits based on meaning rather than character counts or formatting.

**How it works**: Sentences or small segments are embedded, and the cosine similarity between adjacent segments is computed. When the similarity between consecutive segments drops below a threshold (indicating a topic shift), a chunk boundary is placed. The result is chunks that correspond to coherent topics.

**Advantages**: Chunks align with actual topic boundaries. Each chunk is more likely to contain a complete, coherent idea. Retrieval precision improves because chunks are semantically focused.

**Disadvantages**: Requires embedding every sentence during the chunking phase, which adds computational cost and latency to the indexing pipeline. The threshold for "topic shift" must be tuned and varies by domain. Can produce highly variable chunk sizes.

### Document-Structure-Aware Chunking

Uses the document's own structure (headings, sections, HTML tags, Markdown headers, table boundaries) to define chunk boundaries.

**How it works**: Parse the document's structure first. For a Markdown document, identify headers at each level. For HTML, use semantic tags (`<section>`, `<article>`, `<h1>`-`<h6>`). For PDFs, use layout analysis to identify sections, tables, and figures. Then chunk along these structural boundaries, merging small sections and splitting large ones.

**Advantages**: Preserves the author's intended information organization. Section titles can be prepended to chunks as context ("## Financial Results\n\nRevenue grew 12% to..."), dramatically improving both embedding quality and retrieval precision.

**Disadvantages**: Requires document-type-specific parsing. Works beautifully for well-structured documents (technical docs, Wikipedia) but poorly for unstructured text (transcripts, emails, chat logs).

### Specialized Strategies

**Code chunking**: Splits code by functions, classes, or logical blocks using AST (Abstract Syntax Tree) parsing. Ensures that a function is never split mid-body.

**Table chunking**: Tables should generally be kept whole or serialized into text with context ("This table shows quarterly revenue by region:"). Splitting a table row-by-row destroys meaning.

**Agentic/proposition chunking**: Uses an LLM to decompose documents into atomic, self-contained propositions. Each proposition is a standalone fact. This produces the most granular, precise chunks but is expensive (requires an LLM call per document during indexing).

## The Chunk Size Trade-Off

This is the central tension in chunking:

**Too large (e.g., 2000+ tokens)**:
- Chunks contain more noise (irrelevant information mixed with relevant)
- Embedding quality degrades because the vector must represent too many distinct concepts
- Fewer chunks fit in the LLM's context window during generation
- But: context is preserved, and the LLM has more surrounding information to work with

**Too small (e.g., 100 tokens)**:
- Individual chunks lose context and become ambiguous ("It grew by 12%" -- what grew?)
- More chunks must be retrieved to cover a topic, increasing retrieval complexity
- But: retrieval is more precise, and irrelevant information is minimized

**The sweet spot** for most applications is **256-1024 tokens**, with 512 being a common default. However, the optimal size depends on:

- **Query type**: Factual lookups favor smaller chunks; analytical questions favor larger ones
- **Document type**: Dense technical documents can use smaller chunks; narrative text needs larger ones
- **Embedding model**: The embedding model's context window sets an upper bound. Many models are trained on sequences of 512 tokens or less, and performance may degrade on longer inputs even if they technically support them

## Overlap Strategies

Overlap means adjacent chunks share some text at their boundaries. If chunk 1 ends with tokens 490-512 and chunk 2 starts with those same tokens, you have ~10% overlap.

**Why overlap helps**: It ensures that information near chunk boundaries appears in at least one chunk with surrounding context. Without overlap, a fact at the boundary might be split and lost to both chunks.

**Typical overlap**: 10-20% of chunk size. A 512-token chunk with 50-100 tokens of overlap is standard. Higher overlap increases storage requirements but improves boundary handling.

**When to skip overlap**: Semantic chunking and structure-aware chunking split at natural boundaries, making overlap less necessary. Overlap is most important for fixed-size and recursive splitting where boundaries are arbitrary.

## How to Choose for Your Use Case

A practical decision framework:

1. **Start with recursive splitting** at 512 tokens with 50-token overlap. This is the baseline.
2. **Evaluate on representative queries**. Manually inspect what chunks are retrieved for 20-30 real queries. Look for: split facts, missing context, irrelevant noise.
3. **If chunks lack structure context**, switch to document-structure-aware chunking and prepend section headers.
4. **If topic coherence is poor**, experiment with semantic chunking.
5. **If dealing with code**, use AST-based chunking.
6. **Adjust chunk size** based on failure patterns: shrink if noise is the problem, grow if context loss is the problem.
7. **Consider multi-level retrieval**: Create chunks at multiple granularities (small chunks for precise retrieval, parent chunks for context), then retrieve small chunks but inject their parent chunk into the prompt.

## Why It Matters

Chunking is often the most impactful and least glamorous part of a RAG pipeline. Teams spend weeks optimizing embedding models and prompt templates while using default chunking settings -- then discover that fixing their chunking strategy produces bigger improvements than everything else combined. The garbage-in-garbage-out principle applies forcefully: if the retrieval step returns poorly chunked fragments, no amount of downstream sophistication can recover the lost context.

## Key Technical Details

- **Token counting vs. character counting**: Always chunk by tokens (matching your embedding model's tokenizer), not characters. Token-to-character ratios vary by language and content type.
- **Metadata enrichment**: Attach metadata to each chunk -- source document, page number, section title, position in document. This enables filtering and provides context to the LLM.
- **Deduplication**: Overlapping chunks and repeated content across documents create near-duplicate embeddings. Deduplicating at the chunk level improves retrieval diversity.
- **Chunk ID stability**: If you re-chunk a document after edits, design your chunk IDs so that unchanged sections produce the same IDs, avoiding unnecessary re-embedding.

## Common Misconceptions

**"There is one optimal chunk size."** The optimal size depends on the query distribution, document types, embedding model, and downstream task. It must be determined empirically for each application.

**"Semantic chunking is always better than fixed-size."** Semantic chunking adds cost and complexity. For well-structured documents with clear paragraph boundaries, recursive splitting is often equally effective at a fraction of the cost.

**"Chunk size should match the embedding model's max context."** Using the maximum context length often degrades embedding quality. Most embedding models produce better representations for text well within their context limit.

## Connections to Other Concepts

- **RAG** is the primary consumer of chunking -- every RAG pipeline starts with a chunking decision.
- **Embedding models** interact with chunking through their context windows and training characteristics.
- **Vector databases** store chunks and their embeddings; chunk count directly affects index size and cost.
- **Prompt engineering** determines how retrieved chunks are presented to the generative model.
- **Long-context models** change the calculus: larger context windows mean more chunks can be included, partially mitigating imperfect retrieval.

## Further Reading

- Langchain documentation on Text Splitters. The most comprehensive practical guide to different splitting strategies, with code examples for each approach.
- Kamradt, G. (2023). "5 Levels of Text Splitting." A widely referenced analysis progressing from character splitting through semantic chunking, with quality comparisons at each level.
- Chen, J. et al. (2024). "Dense X Retrieval: What Retrieval Granularity Should We Use?" Research comparing different chunk granularities including propositions (atomic facts) and demonstrating that granularity significantly impacts retrieval and generation quality.
