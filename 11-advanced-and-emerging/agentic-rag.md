# Agentic RAG

**One-Line Summary**: Agentic RAG replaces the rigid "retrieve then generate" pipeline with an AI agent that dynamically reasons about what to retrieve, when to retrieve, whether the retrieved information is sufficient, and how to synthesize multi-step retrieval results -- transforming RAG from a fixed pipeline into an adaptive, iterative reasoning process.

**Prerequisites**: Understanding of standard RAG pipelines, AI agents and tool use (function calling), the limitations of single-shot retrieval, ReAct-style reasoning (thought-action-observation loops), and LLM-based planning.

## What Is Agentic RAG?

Standard RAG follows a fixed three-step pipeline: embed query, retrieve top-k documents, generate answer. This works well for simple factual questions where a single retrieval step returns sufficient context. But it breaks down on complex queries that require:

- **Multiple retrieval steps**: "Compare the environmental policies of the EU and China" requires at least two separate retrievals.
- **Adaptive query formulation**: The initial query may not be the right search query. An agent might need to reformulate, decompose, or expand the query based on what it finds (or fails to find).
- **Sufficiency judgment**: The agent needs to decide whether retrieved documents actually answer the question, or whether additional retrieval is needed.
- **Source routing**: Different questions may be best answered by different knowledge bases, APIs, or tools (vector database, SQL database, web search, calculator).
- **Iterative refinement**: Reading the first set of documents may reveal that the question needs to be reinterpreted or broken into sub-questions.

Agentic RAG addresses all of these by wrapping the retrieval process in an agent loop. The LLM is not just the generator -- it is the orchestrator that plans, executes, and evaluates the entire retrieval process.

## How It Works

### Core Architecture

An agentic RAG system consists of:

**1. The Agent (LLM with reasoning capabilities)**: A capable LLM (GPT-4, Claude, etc.) that can reason about what information it needs, formulate retrieval strategies, evaluate results, and decide next steps. The agent operates in a ReAct-style loop: Thought -> Action -> Observation -> Thought -> ...

**2. Tools (retrieval and other capabilities)**: The agent has access to one or more tools:
- `vector_search(query, filters)` -- search the vector database
- `keyword_search(query)` -- BM25 or full-text search
- `web_search(query)` -- internet search
- `sql_query(query)` -- database queries
- `document_reader(doc_id, section)` -- read specific document sections
- `calculator(expression)` -- for numerical reasoning
- Domain-specific APIs

**3. Memory/State**: The agent maintains a working memory of what it has retrieved so far, what questions remain unanswered, and what it has learned.

### The Agent Loop

A typical agentic RAG interaction looks like:

```
User: "What was the impact of the 2023 Basel III endgame proposal on US regional banks'
       capital requirements, and how did this compare to European implementation?"

Agent Thought: This is a complex comparative question. I need to:
  1. Retrieve information about the Basel III endgame proposal (US specific)
  2. Find data on its impact on regional banks' capital requirements
  3. Retrieve information about European Basel III implementation
  4. Compare the two

Action: vector_search("Basel III endgame proposal 2023 US regional banks capital requirements")
Observation: [Retrieved 5 chunks about the proposal's details and capital impact estimates]

Agent Thought: I have good information about the US proposal. The chunks mention
  specific capital requirement increases of 16-20% for large regional banks. Now I
  need European implementation details.

Action: vector_search("Basel III implementation Europe CRR3 capital requirements banks")
Observation: [Retrieved 4 chunks about European CRR3 regulation]

Agent Thought: The European results mention CRR3 but focus on G-SIBs, not regional
  equivalents. Let me search more specifically.

Action: vector_search("CRR3 European medium-sized banks capital impact comparison Basel")
Observation: [Retrieved 3 more targeted chunks]

Agent Thought: I now have sufficient information on both sides. Let me also check if
  there were any revisions to the US proposal.

Action: vector_search("Basel III endgame revised proposal 2024 changes")
Observation: [Retrieved 2 chunks about the September 2024 revisions]

Agent Thought: I have comprehensive information to synthesize a comparative answer.
  Let me generate the final response.

[Generates detailed comparative answer citing specific retrieved documents]
```

### Design Patterns in Agentic RAG

**Router Pattern**: The agent decides which tool or knowledge base to query based on the question type. Financial questions go to the financial database, legal questions to the legal corpus, current events to web search. The routing decision is made by the LLM based on the query semantics.

**Iterative Retrieval Pattern**: The agent retrieves, evaluates, and retrieves again until it has sufficient context. Each retrieval informs the next query. This is particularly powerful for exploratory questions where the user's intent unfolds as information is gathered.

**Sub-question Decomposition Pattern**: Complex queries are broken into atomic sub-questions, each answered through separate retrieval chains. Results are synthesized at the end. For example, "What are the pros and cons of microservices vs. monoliths for a 10-person team?" might be decomposed into separate retrievals about microservice benefits, microservice drawbacks, monolith benefits, monolith drawbacks, and team-size considerations.

**Self-RAG / Reflective Pattern**: After generating an answer, the agent critically evaluates whether the answer is fully supported by the retrieved evidence. If it detects unsupported claims or gaps, it triggers additional retrieval to fill those gaps before producing the final response.

**Multi-Index Routing**: The agent queries multiple specialized indexes (each covering a different domain or document type) and merges results. A customer support agent might query a product documentation index, a troubleshooting guide index, and a customer history database.

### Frameworks and Implementation

**LangGraph**: Provides a graph-based framework for building agentic RAG workflows. Retrieval, evaluation, and generation steps are nodes in a directed graph, with conditional edges that enable branching (retry retrieval, route to different sources, etc.).

**LlamaIndex**: Offers agent abstractions specifically designed for RAG, including SubQuestionQueryEngine (decomposes queries), RouterQueryEngine (routes to appropriate indexes), and tool-augmented agents that can combine retrieval with other actions.

**DSPy**: Treats the entire agentic RAG pipeline as an optimizable program, automatically tuning prompts, few-shot examples, and retrieval strategies through compilation.

**Custom implementations**: Many production systems build custom agent loops using tool-calling APIs (OpenAI function calling, Claude tool use) with explicit state management and error handling.

## Why It Matters

Agentic RAG represents the maturation of RAG from a demo-ready technique to a production-grade architecture. The key insight is that the hard problems in RAG are not in any individual component (embedding, retrieval, generation) but in the orchestration between them. A fixed pipeline cannot handle the diversity of real-world queries:

- **40-60% of real queries are multi-faceted**, requiring multiple retrieval steps or different knowledge sources. Simple RAG handles single-faceted questions well but fails on complex ones.
- **Query-retrieval mismatch** is common: the user's natural language query is often not the best search query. Agentic systems reformulate queries based on retrieved context, closing this gap iteratively.
- **Retrieval failures are invisible in static pipelines**: If the top-k results are irrelevant, standard RAG generates an answer anyway, potentially hallucinating. Agentic RAG evaluates retrieval quality and takes corrective action.

Production deployments at companies building knowledge assistants, research tools, and customer support systems increasingly use agentic RAG patterns because they handle the "long tail" of complex queries that static RAG pipelines cannot address.

## Key Technical Details

- **Token budget management**: Agentic RAG can consume many tokens across multiple LLM calls (planning, tool calls, evaluation, synthesis). Effective implementations budget context window usage, summarize intermediate results, and avoid redundant retrieval.
- **Latency**: Multi-step retrieval adds latency. A standard RAG query takes 1-3 seconds. An agentic RAG query with 3-4 retrieval steps might take 5-15 seconds. Streaming the agent's intermediate reasoning to the user helps manage perceived latency.
- **Reliability**: More agent steps mean more potential failure points. Production agentic RAG systems need robust error handling, retry logic, timeout management, and fallback strategies (e.g., fall back to simple RAG if the agent loop fails).
- **Evaluation**: Evaluating agentic RAG is harder than evaluating standard RAG. You need to assess not just final answer quality but also retrieval strategy quality, query decomposition quality, and sufficiency judgment accuracy.
- **Guardrails**: Without limits, agents can enter infinite retrieval loops or make dozens of unnecessary tool calls. Maximum step counts, relevance thresholds, and cost budgets are essential guardrails.
- **Observability**: Logging every agent thought, action, and observation is critical for debugging and improving the system. Tools like LangSmith, Phoenix (Arize), and custom logging provide this observability.

## Common Misconceptions

**"Agentic RAG always outperforms simple RAG."** For simple factual questions, agentic RAG adds unnecessary latency and cost. The ideal system routes simple queries through a fast standard RAG pipeline and only engages the agent loop for complex queries that need it. This routing decision can itself be made by a lightweight classifier or the agent.

**"More retrieval steps always improve quality."** There are diminishing returns. After 3-5 focused retrieval steps, additional retrievals typically add noise rather than new information. Effective agents know when to stop.

**"You need GPT-4 class models for the agent."** While more capable models make better agents, practical agentic RAG systems can use smaller models (GPT-3.5, Claude Haiku, open-source 7B-13B models) for simpler decisions (routing, sufficiency checking) and reserve large models for complex reasoning and synthesis.

**"Agentic RAG is just prompt chaining."** Agentic RAG involves dynamic decision-making at each step. The agent chooses its next action based on previous observations, which makes the execution path non-deterministic and adaptive. Prompt chaining follows a fixed sequence of LLM calls.

## Connections to Other Concepts

- **RAG**: Agentic RAG is the evolution of standard RAG, adding dynamic orchestration around the same core retrieve-and-generate pattern.
- **AI agents and tool use**: Agentic RAG is a specific application of the general AI agent pattern, where the primary "tools" are retrieval interfaces.
- **Corrective RAG (CRAG)**: CRAG is a specific agentic RAG pattern focused on evaluating and correcting retrieval failures.
- **HyDE**: An agentic RAG system might choose to use HyDE for certain types of queries where direct query embedding is likely to fail.
- **Function calling**: The tool-calling interface of modern LLMs is the mechanism through which agentic RAG systems invoke retrieval and other tools.
- **Compound AI systems**: Agentic RAG is a prime example of compound AI systems -- multiple components (LLM, retriever, evaluator, tools) working together.

## Diagrams and Visualizations

*Recommended visual: Agentic RAG loop showing dynamic retrieval decisions, query reformulation, and iterative refinement — see [LangChain Agentic RAG Documentation](https://python.langchain.com/docs/tutorials/qa_chat_history/)*

*Recommended visual: Comparison of naive RAG pipeline vs agentic RAG with decision points — see [LlamaIndex Agentic RAG](https://docs.llamaindex.ai/en/stable/)*

## Further Reading

- Yao, S. et al. (2023). "ReAct: Synergizing Reasoning and Acting in Language Models." *ICLR 2023.* The foundational framework for interleaving reasoning with actions that underlies most agentic RAG architectures.
- Asai, A. et al. (2024). "Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection." *ICLR 2024.* Trains an LLM to adaptively retrieve and self-evaluate, a key agentic RAG capability.
- Khattab, O. et al. (2023). "DSPy: Compiling Declarative Language Model Calls into Self-Improving Pipelines." Provides a programming framework for building and optimizing agentic RAG systems.
- LangGraph documentation. Practical framework for building agentic RAG with graph-based workflows, conditional routing, and state management.
