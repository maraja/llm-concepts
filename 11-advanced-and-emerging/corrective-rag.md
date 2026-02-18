# Corrective RAG (CRAG)

**One-Line Summary**: Corrective RAG adds a critical evaluation step after retrieval to assess whether retrieved documents are actually relevant to the query, then takes corrective actions -- query rewriting, web search fallback, or knowledge refinement -- when retrieval quality is insufficient, preventing the generation phase from hallucinating over irrelevant context.

**Prerequisites**: Understanding of standard RAG pipelines and their failure modes, the concept of retrieval precision (are the retrieved documents actually relevant?), basic familiarity with agentic RAG patterns, and awareness that LLMs will often confidently generate answers from irrelevant context.

## What Is Corrective RAG?

Standard RAG has a critical blind spot: it assumes the retrieval step succeeds. After retrieving the top-k documents, the pipeline feeds them directly to the LLM for generation, regardless of whether those documents actually contain relevant information. If the query is "What is the company's parental leave policy?" and the retrieval returns chunks about general company benefits, vacation policy, and office locations (because the parental leave document was not well-indexed, or the embedding model failed to match the query), the LLM receives irrelevant context and must either hallucinate an answer or correctly identify that the context does not contain the answer. In practice, LLMs often hallucinate.

Corrective RAG (CRAG), formalized by Yan et al. (2024), addresses this by inserting an explicit evaluation and correction layer between retrieval and generation. A lightweight evaluator (which can be an LLM, a specialized classifier, or a heuristic) assesses the relevance of each retrieved document. Based on this assessment, the system takes one of several corrective actions:

- **Correct**: Retrieved documents are relevant. Proceed to generation normally, potentially after refining the documents to extract the most pertinent information.
- **Incorrect**: Retrieved documents are irrelevant. Trigger alternative retrieval strategies -- web search, query rewriting, searching a different index, or falling back to the LLM's parametric knowledge.
- **Ambiguous**: Some documents are relevant, others are not. Filter out irrelevant documents, possibly supplementing with additional retrieval.

## How It Works

### The CRAG Pipeline

```
User Query -> Initial Retrieval -> Relevance Evaluation -> [Decision Branch]
  |                                                           |
  |-- CORRECT: Knowledge Refinement -> Generation            |
  |-- INCORRECT: Web Search / Query Rewrite -> Retrieval -> Generation
  |-- AMBIGUOUS: Filter + Supplement -> Generation
```

### Step 1: Initial Retrieval

Standard dense retrieval: embed the query, search the vector database, retrieve top-k documents. This is identical to naive RAG.

### Step 2: Relevance Evaluation

Each retrieved document is evaluated for relevance to the query. The CRAG paper proposes training a lightweight evaluator model (T5-large scale, ~770M parameters) that takes a (query, document) pair and outputs a confidence score. The score is mapped to three categories:

- **Correct** (confidence > upper threshold): The document is relevant to the query.
- **Incorrect** (confidence < lower threshold): The document is not relevant.
- **Ambiguous** (between thresholds): Uncertain -- the document may contain some relevant information.

Alternative evaluation approaches include:
- **LLM-as-judge**: Ask the same or a smaller LLM "Is this document relevant to the query?" with a structured output format (yes/no/partial + confidence).
- **Cross-encoder scoring**: Use a cross-encoder reranker model and apply a relevance threshold. Documents scoring below the threshold are deemed irrelevant.
- **Heuristic approaches**: Combine retrieval score, keyword overlap, and metadata signals to estimate relevance without additional model inference.

### Step 3: Corrective Actions

**When evaluation is "Correct"**: The system applies knowledge refinement -- decomposing each relevant document into fine-grained "knowledge strips" (individual sentences or propositions), evaluating each strip's relevance, and retaining only the most pertinent information. This reduces noise and focuses the LLM on the exact evidence that answers the query.

**When evaluation is "Incorrect"**: The system triggers fallback retrieval:

1. **Web search**: Issue the query (possibly reformulated) to a web search API. Web search often succeeds where vector search fails because it has access to a broader knowledge base and uses different matching algorithms.
2. **Query rewriting**: Use an LLM to reformulate the query, then re-run retrieval against the original vector database. The reformulated query may use different terms or decompose a complex query into simpler sub-queries.
3. **Alternative index search**: If multiple indexes are available (e.g., keyword index, graph database, structured database), try a different retrieval modality.
4. **Parametric knowledge fallback**: In some cases, if no retrieval succeeds, acknowledge the limitation or fall back to the LLM's internal knowledge with appropriate caveats.

**When evaluation is "Ambiguous"**: The system combines the partially relevant retrieved documents (after filtering irrelevant ones and refining the relevant ones) with supplementary information from web search or alternative retrieval.

### Step 4: Generation

The refined, corrected context is passed to the generative LLM. Because the context has been vetted for relevance and refined for precision, the generation step produces higher-quality, more faithful outputs.

### CRAG in Graph-Based Workflows

The LangGraph implementation of CRAG has become particularly popular. The workflow is represented as a state graph:

```python
# Simplified CRAG graph structure
graph = StateGraph(State)
graph.add_node("retrieve", retrieve_documents)
graph.add_node("grade_documents", grade_document_relevance)
graph.add_node("generate", generate_answer)
graph.add_node("rewrite_query", rewrite_query)
graph.add_node("web_search", web_search_fallback)

graph.add_edge("retrieve", "grade_documents")
graph.add_conditional_edges(
    "grade_documents",
    decide_next_step,  # Routes based on grading results
    {
        "relevant": "generate",
        "not_relevant": "rewrite_query",
        "partially_relevant": "web_search",
    }
)
graph.add_edge("rewrite_query", "retrieve")  # Re-retrieval loop
graph.add_edge("web_search", "generate")
```

This graph-based representation makes the control flow explicit, debuggable, and modifiable.

## Why It Matters

CRAG addresses the silent failure mode that plagues production RAG systems. In standard RAG, when retrieval fails, the failure is invisible -- the system generates an answer regardless, often a plausible-sounding hallucination. Users cannot distinguish between answers grounded in relevant evidence and answers fabricated from irrelevant context. This erodes trust and can have serious consequences in high-stakes domains (medical, legal, financial).

Key improvements from CRAG:

- **Reduced hallucination from irrelevant context**: By filtering out irrelevant documents before generation, the LLM is far less likely to hallucinate based on tangentially related information.
- **Graceful fallback**: When the primary retrieval fails, the system actively seeks alternative information sources rather than proceeding with bad context.
- **Transparency**: The evaluation step provides an audit trail. You can log and monitor which queries trigger corrective actions, revealing gaps in your knowledge base that need to be filled.
- **Robustness**: CRAG makes RAG systems robust to the inevitable imperfections of embedding models, chunking strategies, and user query formulations.

In the CRAG paper's experiments, the approach improved RAG performance by 5-10% on standard QA benchmarks, with the most dramatic improvements on queries where standard retrieval returned irrelevant documents (up to 20% improvement on those cases).

## Key Technical Details

- **Evaluation latency**: The relevance evaluation step adds 50-500ms per query depending on the evaluator (heuristic < small classifier < cross-encoder < LLM-as-judge). For most applications, this is acceptable given the quality improvement.
- **Threshold tuning**: The upper and lower confidence thresholds that separate "correct," "ambiguous," and "incorrect" must be tuned empirically on a validation set. Too strict thresholds trigger unnecessary web search; too lenient thresholds let irrelevant documents through.
- **Knowledge refinement granularity**: Decomposing documents into individual sentences or propositions and evaluating each one is expensive but effective. In practice, a balance is struck by applying refinement only to longer documents and only when the overall document relevance is borderline.
- **Web search integration**: The web search fallback requires an API (Tavily, Serper, Brave Search, or Google Custom Search). The search results also need to be processed (scraped, cleaned, chunked) before being used as context. Caching frequent web search queries helps with latency and cost.
- **Loop limits**: Query rewriting that triggers re-retrieval can theoretically loop indefinitely. A maximum iteration count (typically 2-3 retries) prevents infinite loops.
- **Cost**: CRAG adds cost through the evaluation step and potential fallback retrieval. For most applications, the improvement in answer quality justifies the 20-50% increase in per-query cost.

## Common Misconceptions

**"CRAG requires training a custom evaluator."** While the original paper trains a T5-based evaluator, practical implementations often use off-the-shelf cross-encoder rerankers with a relevance threshold, or LLM-as-judge prompts. Custom training improves performance but is not strictly necessary.

**"CRAG is a separate system from RAG."** CRAG is a pattern layered on top of standard RAG. It enhances the pipeline with evaluation and correction, but the core retrieve-and-generate architecture remains. Any existing RAG system can be upgraded to CRAG by adding the evaluation and correction layers.

**"Web search fallback means the knowledge base is inadequate."** Web search is a safety net, not an admission of failure. Even excellent knowledge bases have gaps, and user queries can be unpredictable. The fallback ensures the system remains useful even for queries that fall outside the knowledge base's coverage.

**"CRAG solves all retrieval failures."** CRAG handles the case where retrieval returns documents that are irrelevant. It does not solve all RAG failure modes -- for example, it does not address cases where the retrieved documents are relevant but the LLM misinterprets them, or where the answer requires multi-hop reasoning across documents.

## Connections to Other Concepts

- **RAG**: CRAG is an enhancement pattern for standard RAG, specifically addressing retrieval quality evaluation and correction.
- **Agentic RAG**: CRAG is one specific pattern within the broader agentic RAG framework. Agentic RAG encompasses any dynamic retrieval behavior; CRAG specifically focuses on retrieval evaluation and correction.
- **HyDE**: CRAG's query rewriting step can incorporate HyDE -- generating a hypothetical answer as the rewritten query for re-retrieval.
- **ColBERT / reranking**: Cross-encoder rerankers can serve as the relevance evaluator in CRAG, providing calibrated relevance scores.
- **Guardrails and safety**: CRAG shares the philosophy of evaluation-before-action with guardrails systems -- both add a checking layer to prevent poor outputs.
- **Hallucination**: CRAG specifically targets context-based hallucination (the LLM generating answers from irrelevant context), one of the primary hallucination modes in RAG systems.

## Further Reading

- Yan, S. et al. (2024). "Corrective Retrieval Augmented Generation." (arXiv: 2401.15884) The original CRAG paper introducing the retrieval evaluator and corrective action framework.
- Asai, A. et al. (2024). "Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection." A related approach where the LLM itself learns to generate retrieval and critique tokens, enabling self-corrective behavior.
- LangGraph documentation: "Corrective RAG." Practical implementation of CRAG as a LangGraph workflow with code examples.
- Gao, Y. et al. (2024). "Retrieval-Augmented Generation for Large Language Models: A Survey." Places CRAG in the broader taxonomy of advanced RAG techniques.
