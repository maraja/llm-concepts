# Compound AI Systems

**One-Line Summary**: Compound AI systems combine LLMs with retrievers, tools, code execution, verifiers, and other models into integrated architectures that exceed the capabilities of any single model, representing the shift from "better models" to "better systems" as the primary path to improved AI performance.

**Prerequisites**: Understanding of LLM capabilities and limitations, retrieval-augmented generation (RAG), API design, software engineering principles, the concept of tool use in AI, and basic systems thinking (components, interfaces, failure modes).

## What Are Compound AI Systems?

Consider the difference between a brilliant individual and a well-organized team. A single brilliant person has limits: they cannot look up every fact from memory, perform perfect calculations in their head, or be an expert in every domain simultaneously. But that same person, given access to a library, a calculator, specialized colleagues, and a quality-checking process, becomes far more capable. Compound AI systems apply this same principle to LLMs.

*Recommended visual: Compound AI system architecture combining LLM, retriever, code executor, and verifier components — see [Berkeley AI Research Blog – The Shift from Models to Compound AI Systems](https://bair.berkeley.edu/blog/2024/02/18/compound-ai-systems/)*


A compound AI system is an architecture that combines multiple interacting components -- at least one of which is typically an LLM -- to accomplish tasks that no single component could handle reliably alone. The term was popularized by the Berkeley AI Research (BAIR) group in 2024, capturing a shift already underway across the industry: the realization that improving the **system around the model** often yields more practical gains than improving the model itself.

ChatGPT with Code Interpreter, Claude with tool use and computer use, and Perplexity's search-augmented generation are all compound AI systems. They combine a language model's reasoning with grounded capabilities (code execution, web search, document retrieval) to produce outputs that are more accurate, verifiable, and useful than what the LLM alone could generate.

## How It Works


*Recommended visual: DSPy programming framework for optimizing compound AI systems — see [DSPy Paper (arXiv:2310.03714)](https://arxiv.org/abs/2310.03714)*

### Core Design Patterns

**1. Retrieval-Augmented Generation (RAG)**

The most widespread compound pattern. Instead of relying on the model's parametric memory:

```
Query -> Retriever (search index) -> Relevant Documents -> LLM (generate answer grounded in documents) -> Response
```

Components: embedding model (for query/document encoding), vector database (for efficient similarity search), reranker (for refining retrieved results), and the generator LLM. The retriever provides factual grounding; the LLM provides reasoning and synthesis.

**2. Verified Generation**

Pair a generator with a verifier to catch and correct errors:

```
Query -> Generator LLM -> Draft Response -> Verifier (code execution, fact-checker, logic checker) -> Feedback -> Revised Response
```

Example: An LLM generates code, a sandbox executes it against test cases, and failures are fed back for correction. The verification loop can iterate multiple times. This is the principle behind coding agents that achieve high scores on SWE-bench.

**3. Tool-Augmented Reasoning**

The LLM decides when and how to invoke external tools:

```
Query -> LLM (reason about approach) -> Tool Call (calculator, API, database, web search) -> Tool Result -> LLM (incorporate result, continue reasoning) -> Response
```

The LLM acts as an orchestrator, decomposing problems into steps that combine its reasoning with tool capabilities. Function calling (structured tool invocation) is the interface mechanism.

**4. Multi-Agent Collaboration**

Multiple LLM instances play different roles:

```
User Query -> Planner Agent (decompose task) -> [
  Researcher Agent (gather information),
  Coder Agent (write and test code),
  Critic Agent (review and identify issues)
] -> Synthesizer Agent (combine results) -> Response
```

Each agent may use different models, prompts, or tools. The collaboration structure can be hierarchical (supervisor-worker), peer-to-peer, or debate-style (agents argue for different positions).

**5. Router/Cascade Architecture**

Direct queries to the most appropriate handler:

```
Query -> Router (classify difficulty/type) -> [
  Simple queries: Small, fast model
  Complex queries: Large, capable model
  Specialized queries: Domain-specific model
  Factual queries: RAG pipeline
] -> Response
```

This optimizes the cost-quality tradeoff: most queries are easy and can be handled cheaply, while hard queries get more resources.

### The Engineering Discipline

Building reliable compound systems requires addressing challenges that do not exist for single models:

- **Error propagation**: If the retriever returns irrelevant documents, the generator produces a confidently wrong answer grounded in bad evidence. Each component's failure mode must be analyzed and mitigated.
- **Latency management**: Sequential tool calls add latency. Parallelizing independent operations, caching intermediate results, and speculative execution are essential.
- **Observability**: Debugging a compound system requires tracing decisions across components. Logging intermediate states (what was retrieved, what tools were called, what the verifier found) is critical.
- **Evaluation**: End-to-end metrics may not reveal which component is failing. Component-level evaluation (retrieval precision, tool accuracy, generation quality) is needed alongside system-level metrics.

## Why It Matters

The compound systems paradigm represents a fundamental reorientation of the AI field:

- **Practical capability leap**: ChatGPT became vastly more useful when it could execute code, search the web, and process files. The model did not change -- the system around it did.
- **Grounding and reliability**: RAG reduces hallucination. Code execution verifies correctness. Fact-checking catches errors. Systems can provide guarantees that individual models cannot.
- **Modularity and upgradeability**: When a better retriever or a better model becomes available, it can be swapped in without redesigning the entire system. Components are independently improvable.
- **Cost optimization**: Routers and cascades ensure expensive models are only used when necessary, dramatically reducing average inference cost while maintaining quality.
- **Capabilities beyond any single model**: No single model can simultaneously have perfect factual recall, execute arbitrary code, access real-time information, and interact with external services. Systems compose these capabilities.

## Key Technical Details

- **Function calling protocol**: Most compound systems use structured function calling where the LLM outputs a JSON-formatted tool invocation, the system executes the tool, and the result is fed back to the LLM. OpenAI, Anthropic, and open-source frameworks all converge on this pattern.
- **Orchestration frameworks**: LangChain, LlamaIndex, CrewAI, AutoGen, and similar frameworks provide abstractions for building compound systems. They handle tool routing, context management, and agent coordination.
- **Retrieval quality is the bottleneck**: In RAG systems, the retriever's precision is often the weakest link. Two-stage retrieval (fast embedding search followed by reranking) and query reformulation significantly improve results.
- **Context window management**: Each tool result, retrieved document, and intermediate reasoning step consumes context tokens. Effective systems manage this budget carefully, summarizing or truncating as needed.
- **Guardrails and safety layers**: Production compound systems include input classifiers (detecting harmful queries), output filters (blocking unsafe content), and policy enforcement -- additional components wrapping the core pipeline.
- **Determinism and reproducibility**: Compound systems with multiple stochastic components (LLM sampling, retrieval randomness) can produce different outputs for identical inputs. Testing requires statistical evaluation over many runs.

## Common Misconceptions

- **"Compound systems are just prompt engineering"**: Prompt engineering operates within a single model call. Compound systems involve multiple components, external data sources, code execution, and multi-step workflows. The engineering challenges are fundamentally different -- distributed systems concerns like reliability, latency, error handling, and observability dominate.
- **"Better models will eliminate the need for compound systems"**: Even the most capable models benefit from tools, retrieval, and verification. A perfect model still cannot access real-time information without search, execute code without a sandbox, or guarantee factual accuracy without verification. The compound approach and model improvement are complementary.
- **"Multi-agent systems are always better"**: Multiple agents add complexity, latency, and failure modes. For many tasks, a single well-prompted model with good tools outperforms a multi-agent architecture. Agents should be introduced when task decomposition genuinely requires different capabilities or perspectives.
- **"RAG solves hallucination"**: RAG reduces but does not eliminate hallucination. Models can still ignore retrieved documents, misinterpret them, or hallucinate claims not present in the evidence. Verification layers are still needed.
- **"More tools means better performance"**: Giving a model too many tools creates a selection problem. The model may choose the wrong tool, waste tokens on unnecessary tool calls, or become confused by overlapping functionality. Curated, well-documented tool sets outperform large unfocused ones.

## Connections to Other Concepts

- **RAG (Retrieval-Augmented Generation)**: The most fundamental compound pattern. Understanding embedding-based retrieval, vector databases, and reranking is essential.
- **Tool Use and Function Calling**: The interface mechanism that allows LLMs to interact with external systems. Understanding structured output generation and tool selection is crucial.
- **Test-Time Compute**: Many test-time compute techniques (sampling, verification, search) are compound system patterns. Reasoning models internally implement verified generation loops.
- **Evaluation**: Compound systems require multi-level evaluation -- component-level, integration-level, and end-to-end -- making evaluation significantly more complex.
- **AI Safety and Guardrails**: Production compound systems incorporate safety as additional system components (input/output filters, content classifiers, policy enforcement).
- **Context Window Extension**: Longer contexts enable richer compound interactions -- more retrieved documents, longer tool outputs, more conversation history -- but context management remains essential even with large windows.
- **Multimodal Models**: Multimodal models are themselves compound (encoder + projector + LLM) and increasingly serve as components in larger multimodal compound systems.

## Further Reading

- **"The Shift from Models to Compound AI Systems" (Zaharia et al., BAIR Blog, 2024)**: The influential blog post that named and framed the paradigm shift, arguing that systems engineering is becoming as important as model training.
- **"Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks" (Lewis et al., 2020)**: The foundational RAG paper establishing the pattern of combining retrieval with generation.
- **"Toolformer: Language Models Can Teach Themselves to Use Tools" (Schick et al., 2023)**: Demonstrates that LLMs can learn to use tools (calculators, search engines, calendars) through self-supervised learning, enabling autonomous tool selection.
