# ReAct Pattern (Reasoning + Acting)

**One-Line Summary**: ReAct interleaves chain-of-thought reasoning with tool-calling actions in a unified Thought-Action-Observation loop, grounding LLM reasoning in real-world feedback.

**Prerequisites**: Prompt engineering, chain-of-thought prompting, tool use and function calling

## What Is the ReAct Pattern?

Imagine you are solving a crossword puzzle. You do not just think silently and write all answers at once, nor do you randomly try letters without thinking. Instead, you reason ("5-across is probably 'TORCH' because the clue says 'portable light'"), act (write the letters), observe (check if the crossing words still make sense), and adjust. This interleaving of thinking, doing, and observing is exactly what ReAct brings to LLM agents.

Before ReAct, there were two separate paradigms. Chain-of-thought (CoT) prompting let models reason step-by-step but entirely inside their own heads -- with no way to check facts or interact with the world. This led to confident hallucinations where the model would fabricate plausible-sounding but incorrect facts. Conversely, action-only agents (Act) could call tools and APIs but had no explicit reasoning about when or why to use them, leading to poor planning, wasted actions, and inefficient tool usage.

ReAct unifies both: the model's reasoning explicitly plans tool use, and the observations from tool use ground the model's reasoning in reality. Introduced by Yao et al. at Princeton and Google Brain (ICLR 2023), ReAct has become the foundational pattern behind virtually every modern agent framework. Whether you are using LangChain's AgentExecutor, OpenAI's function calling, or Anthropic's tool use, the underlying loop is a descendant of ReAct.

## How It Works

### The Thought-Action-Observation Loop

ReAct operates in a simple iterative cycle:

```
Loop until task is complete or max iterations reached:
  1. Thought: The model reasons about the current state, what it knows,
     what it still needs, and what action to take next.
  2. Action: The model selects and invokes a tool (search, calculator,
     code interpreter, API call, etc.) with specific parameters.
  3. Observation: The tool returns results, which are appended to the
     context for the next iteration of the loop.
```

A concrete example for multi-hop question answering:

```
Question: What is the population of the capital of France?

Thought 1: I need to find the capital of France first, then look up
           its population. Let me search for the capital.
Action 1: Search["capital of France"]
Observation 1: The capital of France is Paris.

Thought 2: The capital is Paris. Now I need the current population.
Action 2: Search["population of Paris 2024"]
Observation 2: The population of Paris is approximately 2.1 million
               (city proper) as of the 2024 census estimate.

Thought 3: I now have both pieces of information. The capital of
           France is Paris with a population of ~2.1 million.
Action 3: Finish["The capital of France is Paris, with a population
          of approximately 2.1 million (city proper)."]
```

The key is that each Thought explicitly references what was learned from the previous Observation and plans the next Action. This creates a traceable reasoning chain grounded in external evidence.

### ReAct vs CoT vs Act-Only: A Comparison

| Dimension           | CoT Only        | Act Only        | ReAct             |
|---------------------|-----------------|-----------------|-------------------|
| Reasoning           | Explicit        | None            | Explicit          |
| Tool use            | None            | Yes             | Yes               |
| Grounding           | None (internal) | Tool outputs    | Tool outputs      |
| Hallucination risk  | High            | Medium          | Low               |
| Planning quality    | Good            | Poor            | Good              |
| Interpretability    | High            | Low             | High              |
| Token cost          | Low             | Medium          | High              |

CoT excels at pure reasoning but hallucinates facts. Act-only uses tools but without strategic planning. ReAct combines the strengths of both while mitigating their weaknesses, at the cost of higher token usage per query.

### Benchmark Results

The original paper demonstrated clear improvements across multiple domains:

- **ALFWorld** (interactive text environment): ReAct achieved 71% success versus 45% for Act-only. Reasoning helped plan multi-step household tasks like "find a cup, rinse it, place it on the counter."
- **FEVER** (fact verification): ReAct scored 60.9% versus 56.3% for CoT-only. Tool use grounded factual claims in actual Wikipedia evidence rather than relying on parametric memory.
- **HotpotQA** (multi-hop QA): ReAct outperformed CoT by leveraging Wikipedia search to retrieve facts rather than hallucinating answers to questions requiring information from multiple documents.

The improvements are most dramatic on tasks requiring both factual knowledge retrieval and multi-step reasoning -- precisely the tasks where neither CoT nor Act alone is sufficient.

### How Modern Frameworks Implement ReAct

The original ReAct paper used free-text parsing of Thought/Action/Observation strings. Modern frameworks have evolved this:

- **LangChain AgentExecutor**: Wraps the ReAct loop with structured tool definitions and output parsers. The agent selects tools from a registry.
- **OpenAI function calling**: Replaces free-text action parsing with structured JSON function calls, eliminating parsing errors entirely.
- **Anthropic tool use**: Similar to OpenAI's approach, with XML-structured tool invocations integrated into the model's generation.

All of these are architecturally ReAct -- they just use cleaner interfaces than raw text parsing for the Action step.

## Why It Matters

1. **Foundation of modern agents**: Every major agent framework implements some variant of the ReAct loop -- it is the conceptual backbone of LLM-powered autonomous agents.
2. **Reduces hallucination**: By grounding reasoning in tool observations, ReAct prevents the confident fabrication that plagues pure chain-of-thought approaches on knowledge-intensive tasks.
3. **Interpretable decision-making**: The explicit Thought steps create a human-readable audit trail of the agent's reasoning, making debugging, monitoring, and compliance straightforward.
4. **Flexible tool integration**: ReAct is tool-agnostic -- the same pattern works with search engines, calculators, databases, APIs, code executors, file systems, or any other callable tool.
5. **Composability**: ReAct agents can be composed into multi-agent systems where each agent runs its own ReAct loop while collaborating with other agents through message passing.

## Key Technical Details

- ReAct prompts are typically few-shot, with 3-6 examples showing the Thought-Action-Observation format for the target domain
- The action space must be explicitly defined; common actions include Search, Lookup, Calculate, CodeExecute, and Finish
- Token cost is 2-4x higher than single-pass generation due to the iterative loop and observation text appended to context
- Most implementations cap the loop at 5-15 iterations to prevent runaway execution and cost explosion
- ReAct can be combined with self-consistency (sample multiple reasoning paths and majority-vote) for further accuracy gains
- Modern implementations replace free-text action parsing with structured function calling (JSON schema), reducing parsing errors to near zero
- The pattern degrades gracefully: if tools are unavailable or return errors, the Thought steps still provide chain-of-thought reasoning as a fallback
- Observation truncation is important: raw tool outputs (e.g., full web pages) must be summarized or truncated to avoid context window overflow
- Temperature settings matter: lower temperature (0.0-0.3) for action selection, optionally higher for creative reasoning steps

## Common Misconceptions

- **"ReAct is just chain-of-thought with tools bolted on."** The integration is deeper than that. In ReAct, reasoning explicitly plans actions, and observations explicitly update reasoning. Simply appending tool outputs to a CoT prompt without this tight coupling of planning and grounding performs measurably worse in controlled experiments.
- **"ReAct requires special model fine-tuning."** The original paper used few-shot prompting with standard off-the-shelf LLMs. No fine-tuning is needed -- just careful prompt design with Thought-Action-Observation examples that demonstrate the desired reasoning-acting behavior.
- **"ReAct always outperforms chain-of-thought."** On tasks that require no external information (pure logic puzzles, arithmetic, code generation from spec), CoT can match or exceed ReAct at lower cost. ReAct's advantage is specifically in knowledge-intensive and interactive tasks where external grounding is valuable.

## Connections to Other Concepts

- **Chain-of-Thought Prompting**: ReAct extends CoT by adding the Action and Observation steps, grounding the internal reasoning process in external tool feedback.
- **Tool Use and Function Calling**: ReAct provides the reasoning framework for deciding when, why, and how to invoke tools -- it is the "brain" behind tool selection.
- **Self-Reflection**: Self-reflection extends ReAct by adding an evaluative step after task completion, enabling learning across trials rather than just within a single execution.
- **Multi-Agent Systems**: Individual agents within multi-agent frameworks typically run ReAct loops internally while coordinating with other agents at the orchestration layer.

## Diagrams and Visualizations

![ReAct framework showing interleaved Thought-Action-Observation steps grounding reasoning in environment feedback](https://lilianweng.github.io/posts/2023-06-23-agent/agent-overview.png)
*Source: [Lilian Weng – LLM Powered Autonomous Agents](https://lilianweng.github.io/posts/2023-06-23-agent/)*

*Recommended visual: ReAct vs Chain-of-Thought vs Act-Only comparison showing how interleaving reasoning with actions improves performance — see [ReAct Paper (arXiv:2210.03629)](https://arxiv.org/abs/2210.03629)*

## Further Reading

- Yao et al., "ReAct: Synergizing Reasoning and Acting in Language Models," arXiv:2210.03629, ICLR 2023
- Wei et al., "Chain-of-Thought Prompting Elicits Reasoning in Large Language Models," NeurIPS 2022
- Schick et al., "Toolformer: Language Models Can Teach Themselves to Use Tools," arXiv:2302.04761, 2023
- Shinn et al., "Reflexion: Language Agents with Verbal Reinforcement Learning," arXiv:2303.11366, NeurIPS 2023
