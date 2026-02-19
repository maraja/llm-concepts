# HyDE (Hypothetical Document Embeddings)

**One-Line Summary**: HyDE bridges the semantic gap between queries and documents by using an LLM to generate a hypothetical answer document, then embedding that hypothetical document (instead of the original query) as the retrieval vector -- leveraging the insight that a fabricated-but-plausible answer is closer in embedding space to real answers than the question itself is.

**Prerequisites**: Understanding of RAG retrieval pipelines (embed query, search vector database, return top-k), the asymmetry between questions and answers in embedding space, how LLMs generate text, and the concept of dense retrieval with bi-encoders.

## What Is HyDE?

Dense retrieval works by embedding both queries and documents into the same vector space, then finding documents whose embeddings are closest to the query embedding. But there is a fundamental asymmetry: a question and its answer are semantically related but structurally very different. The query "What are the health effects of microplastics?" is a short question. The relevant documents are long passages full of specific facts, chemical names, and study results. The question and the answer occupy different regions of embedding space, and the gap between them is the primary failure mode of naive dense retrieval.

*Recommended visual: HyDE pipeline: query → LLM generates hypothetical answer → embed hypothetical document → retrieve real documents — see [Gao et al. HyDE Paper (arXiv:2212.10496)](https://arxiv.org/abs/2212.10496)*


HyDE, introduced by Gao et al. (2022), addresses this with an elegantly simple idea: before retrieval, ask an LLM to generate a hypothetical document that answers the query. This hypothetical answer is probably imprecise, possibly wrong, and certainly not grounded in your actual knowledge base. But it does not need to be correct. It just needs to look like the kind of document that would answer the query. Then, embed this hypothetical document and use that embedding for retrieval instead of the query embedding.

The hypothetical document's embedding is much closer in vector space to the real relevant documents than the original query's embedding, because it shares the same structure, vocabulary, and topic density as real answers. The vector database then finds actual documents that are similar to this hypothetical answer, and those real documents are used for the final generation step.

## How It Works

### The HyDE Pipeline

```
User Query -> LLM generates hypothetical answer -> Embed hypothetical answer -> Search vector database -> Retrieve real documents -> Generate final answer from real documents
```

**Step 1: Hypothetical Document Generation**

The user's query is sent to an LLM (e.g., GPT-4, Claude, or even a smaller instruction-tuned model) with a prompt like:

```
Please write a passage that would answer the following question.
Question: {user_query}
Passage:
```

The LLM generates a hypothetical passage. For the query "What causes aurora borealis?", the LLM might generate:

> "The aurora borealis, or northern lights, is caused by interactions between the solar wind and Earth's magnetosphere. Charged particles from the sun, primarily electrons and protons, travel along magnetic field lines toward the polar regions. When these particles collide with atmospheric gases -- primarily oxygen and nitrogen -- the gases become ionized and emit photons of light. Oxygen produces green and red light at different altitudes, while nitrogen produces blue and purple hues..."

This passage may contain inaccuracies, but it is structurally and topically very similar to real documents about the aurora borealis.

**Step 2: Embedding the Hypothetical Document**

The hypothetical passage is embedded using the same embedding model used to embed the document corpus. This produces a vector that lives in "document space" rather than "query space."

**Step 3: Retrieval**

The hypothetical document's embedding is used as the query vector for nearest-neighbor search. Because it resembles a real document in structure and content, it retrieves documents that are topically similar -- the actual authoritative passages about the aurora borealis from your knowledge base.

**Step 4: Generation**

The real retrieved documents (not the hypothetical one) are injected into the LLM's prompt for final answer generation, exactly as in standard RAG. The hypothetical document is discarded; it was only an intermediate retrieval probe.

### Why It Works: The Embedding Space Geometry

The effectiveness of HyDE comes from the geometry of embedding spaces. Bi-encoder embedding models are trained to place semantically similar texts close together. But "similar" is ambiguous: a question and its answer are semantically related but not semantically similar in the way two answer passages on the same topic are.

Consider three texts:
- Q: "What causes aurora borealis?"
- D1: "The aurora borealis results from solar wind particles interacting with Earth's magnetosphere..."
- D2: "Northern lights occur when charged solar particles collide with atmospheric gases..."

In embedding space, D1 and D2 are very close (they are paraphrases of the same answer). Q is related to both but further away (it is a question, not an answer). The hypothetical document H generated by HyDE is like a noisy version of D1/D2 -- it is in the same neighborhood of embedding space. So searching with H as the query retrieves D1 and D2 more reliably than searching with Q.

### Variations and Extensions

**Multi-hypothesis HyDE**: Generate multiple hypothetical documents (using temperature > 0 for diversity) and either average their embeddings or run separate searches and merge results. This reduces the impact of any single generation's biases.

**Domain-specific prompting**: Tailor the generation prompt to match the corpus style. For a legal database: "Write a legal memorandum addressing..." For a medical database: "Write a clinical summary describing..." This further reduces the gap between the hypothetical document and real corpus documents.

**HyDE with query decomposition**: For complex queries, first decompose into sub-questions, generate hypothetical answers for each, and retrieve separately. This addresses multi-faceted queries where a single hypothetical document cannot cover all aspects.

**Step-back HyDE**: Instead of generating a direct answer, generate a broader contextual passage. For "What was Apple's revenue in Q3 2023?", generate a passage about Apple's financial performance more broadly, retrieving a wider set of relevant financial documents.

## Why It Matters

HyDE addresses the single biggest failure mode in dense retrieval: the query-document asymmetry gap. In benchmarks, HyDE consistently improves retrieval quality over standard query embedding, particularly for:

- **Keyword-poor queries**: Vague or conversational queries like "that thing with the northern lights" become rich hypothetical documents with specific terminology.
- **Domain-specific jargon**: If the user does not know the technical terms, the LLM's hypothetical answer fills in the vocabulary. "Why do my joints hurt when it rains?" generates a passage mentioning barometric pressure, synovial fluid, and arthritis -- terms that match the medical documents in the corpus.
- **Cross-lingual retrieval**: HyDE can bridge language gaps by generating the hypothetical document in the corpus language rather than the query language.

Gao et al. (2022) showed that HyDE with the unsupervised Contriever model outperformed fine-tuned dense retrievers on several BEIR benchmark tasks, demonstrating that a single LLM generation step can compensate for lack of retrieval-specific training data.

## Key Technical Details

- **Latency cost**: HyDE adds one LLM generation call before retrieval. With a fast model (GPT-3.5-turbo, Claude Haiku, or a local model), this adds 200-1000ms. For applications where retrieval quality is more important than latency, this is an acceptable trade-off. For sub-100ms retrieval requirements, HyDE may be too slow.
- **Cost**: The additional LLM call adds per-query cost. For high-volume applications, this can be significant. Using smaller, cheaper models for the hypothetical generation step helps control costs.
- **Correctness is not required**: The hypothetical document does not need to be factually correct. It needs to occupy the right region of embedding space. Even a hallucinated answer about the aurora borealis will use the right vocabulary and structure to be near real aurora borealis documents. This is counterintuitive but is the key insight.
- **Quality of the LLM matters**: Better LLMs generate more plausible hypothetical documents, which produce more accurate retrieval probes. However, even relatively small instruction-tuned models (7B parameters) can generate useful hypothetical documents for HyDE.
- **Not a replacement for query expansion**: HyDE is complementary to query expansion techniques. You can expand the query, use HyDE, or both. They operate on different aspects of the retrieval problem.
- **Embedding model compatibility**: HyDE works with any embedding model. It does not require special training or modification of the embedding model or the vector database.

## Common Misconceptions

**"HyDE always improves retrieval."** It does not. For precise, keyword-heavy queries (e.g., error codes, specific names, exact phrases), the original query embedding may outperform HyDE because the LLM's hypothetical document introduces noise and dilutes the specific terms. HyDE is most beneficial for semantic, conceptual, or underspecified queries.

**"The hypothetical document needs to be correct."** This is the most common misconception. The hypothetical document is a retrieval probe, not an answer. It just needs to be in the right neighborhood of embedding space. A partially wrong but topically relevant hypothetical document still retrieves the right real documents.

**"HyDE is too expensive for production."** With modern fast inference (GPT-3.5-turbo, Claude Haiku, or local models with vLLM), the latency and cost of generating a short hypothetical passage can be quite manageable. The trade-off depends on query volume and quality requirements.

**"HyDE replaces reranking."** HyDE improves the initial retrieval stage (recall). Reranking improves the ordering of retrieved results (precision). They are complementary and can be combined: HyDE retrieval followed by cross-encoder reranking is a powerful pipeline.

## Connections to Other Concepts

- **RAG**: HyDE is an advanced RAG retrieval technique that improves the query->retrieval phase without changing the indexing or generation phases.
- **Embedding models**: HyDE exploits the structure of embedding spaces, making it sensitive to the quality and characteristics of the embedding model used.
- **Prompt engineering**: The quality of the hypothetical document generation prompt directly impacts HyDE effectiveness.
- **Query decomposition**: HyDE can be combined with query decomposition for complex multi-faceted queries.
- **Agentic RAG**: In agentic RAG systems, the agent can decide dynamically whether to use HyDE or direct query embedding based on query characteristics.
- **Corrective RAG**: If initial HyDE retrieval returns irrelevant documents, CRAG mechanisms can trigger re-retrieval with different strategies.

## Further Reading

- Gao, L. et al. (2022). "Precise Zero-Shot Dense Retrieval without Relevance Labels." (arXiv: 2212.10496) The original HyDE paper, demonstrating effectiveness across BEIR benchmarks without any relevance supervision.
- Gao, Y. et al. (2024). "Retrieval-Augmented Generation for Large Language Models: A Survey." Covers HyDE as one of several advanced retrieval techniques in the broader RAG landscape.
- Mao, Y. et al. (2021). "Generation-Augmented Retrieval." An earlier related approach that uses generation to improve retrieval, providing intellectual context for HyDE.
