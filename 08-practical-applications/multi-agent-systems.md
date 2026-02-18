# Multi-Agent Systems

**One-Line Summary**: Multiple LLM-powered agents collaborate through defined roles, tools, and communication protocols to solve problems that exceed the capability of any single agent.

**Prerequisites**: Prompt engineering, tool use and function calling, ReAct pattern, memory systems

## What Are Multi-Agent Systems?

Think of a multi-agent system like a well-run software engineering team. You have a project manager who breaks down tasks, a developer who writes code, a reviewer who checks quality, and a tester who validates results. No single person does everything -- each brings specialized expertise and they communicate through structured processes. Multi-agent LLM systems work the same way: instead of prompting one model to do everything, you instantiate multiple agents with distinct roles, tools, and instructions, then orchestrate their collaboration.

The key insight is that specialization improves performance. A single monolithic prompt trying to handle research, coding, testing, and documentation simultaneously tends to lose focus and produce mediocre results across all dimensions. By decomposing that into agents with narrow mandates, each agent can maintain a tighter context window focused on its specialty, use role-specific tools without distraction, and be evaluated independently against clear success criteria. The orchestration layer handles routing messages between agents, sequencing their execution, and resolving conflicts when agents disagree.

This approach has exploded in popularity since 2023, with frameworks like AutoGen, CrewAI, and LangGraph each offering different abstractions for building multi-agent workflows. The space is evolving rapidly, but the core patterns are stabilizing around a few well-understood architectures that balance flexibility with reliability.

The practical question is not whether multi-agent systems can work -- they clearly can -- but when the added complexity and cost are justified versus a single capable agent with good tools. Understanding the tradeoffs is essential for making that decision.

## How It Works

### Major Frameworks

**AutoGen (Microsoft)**: The most popular framework with 35K+ GitHub stars as of early 2025. Built around the `ConversableAgent` abstraction -- every agent can send and receive messages. Supports group chat with automatic or manual speaker selection, nested conversations, and teachable agents that learn from human feedback. AutoGen's strength is maximum flexibility: agents can be LLMs, humans, tool executors, or any combination. It also supports code execution in sandboxed Docker containers. A minimal setup looks like:

```python
from autogen import ConversableAgent, GroupChat, GroupChatManager

coder = ConversableAgent("coder", system_message="You write Python code.")
reviewer = ConversableAgent("reviewer", system_message="You review code for bugs.")
manager = GroupChatManager(groupchat=GroupChat(agents=[coder, reviewer]))
```

**CrewAI**: Takes a role-playing philosophy where each agent has a defined role, goal, and backstory that shapes its behavior. The core abstractions are Agent (who does the work), Task (what needs to be done, including expected output format), and Crew (the team assembled for the job). Supports sequential processes (agents execute tasks in a defined order) and hierarchical processes (a manager agent dynamically delegates tasks to workers based on their roles). CrewAI prioritizes simplicity and developer experience over maximum flexibility, making it the easiest framework to get started with.

**LangGraph**: Models agent workflows as state machine graphs with nodes (actions) and edges (transitions). Offers built-in persistence, human-in-the-loop checkpoints, and subgraph composition for modular design. LangGraph's state-machine approach gives the most fine-grained control over execution flow, supports conditional branching and cycles, and integrates with the broader LangChain ecosystem. The tradeoff is a steeper learning curve and more boilerplate code than the other frameworks.

**Others worth noting**:

- **CAMEL**: Pioneered role-playing communication between agents using inception prompting, where agents are instructed to stay in character and collaborate toward a shared goal.
- **MetaGPT**: Assigns software engineering roles (PM, architect, engineer, QA) with structured standard operating procedures (SOPs) that define how each role produces and consumes artifacts.
- **ChatDev**: Simulates an entire software company with agents in a waterfall-style pipeline (design, coding, testing, documentation) for automated end-to-end code generation from natural language requirements.

### Collaboration Patterns

There are four primary patterns for multi-agent collaboration:

1. **Debate/Adversarial**: Agents argue opposing positions, improving factual accuracy through argumentation. Du et al. (2023) showed a +15% improvement on TruthfulQA with multi-agent debate versus single-agent generation. Each agent defends a position and must respond to counterarguments.
2. **Delegation/Hierarchical**: A manager agent decomposes tasks and assigns them to specialist workers. Resembles a tree structure where the root coordinates leaf execution. The manager decides which agent handles which subtask based on their declared capabilities.
3. **Pipeline/Sequential**: Agents process work in stages -- one agent's output becomes the next agent's input. Ideal for workflows like draft, review, revise, or research, outline, write, edit.
4. **Voting/Ensemble**: Multiple agents independently solve the same problem, then a majority vote or aggregation step selects the best answer. Trades cost for reliability on high-stakes decisions.

### Communication Protocols

Agents exchange information through several mechanisms:

- **Natural language messages**: The most flexible approach. Agents converse in plain text, just like humans chatting. Easy to debug but can be ambiguous.
- **Structured JSON payloads**: Agents send typed, schema-validated messages. Reduces parsing errors but requires predefined message formats.
- **Shared blackboard state**: A common data store (Redis, database, in-memory dict) that all agents can read and write. Good for collaborative state but requires concurrency management.
- **Tool-mediated interaction**: One agent invokes another as a callable tool. Provides clean interfaces but limits conversational flexibility.

The choice of communication protocol significantly impacts system behavior. Natural language is the most forgiving but the least reliable for structured data handoffs. JSON payloads are precise but brittle to schema changes. In practice, most production systems use a combination: natural language for high-level coordination and structured formats for data transfer between agents.

### When Multi-Agent Wins (and Loses)

Multi-agent systems excel in three scenarios:

- **Complex multi-tool tasks**: When a workflow requires 5+ different tools across different domains (web search, code execution, database queries, API calls), splitting these across specialized agents reduces per-agent complexity and improves reliability.
- **Error-critical workflows**: When mistakes are costly, adding reviewer and validator agents creates defense-in-depth. A coding agent plus a testing agent plus a security review agent catches more bugs than any single agent.
- **Parallelizable workloads**: When subtasks are independent (researching 10 different topics, analyzing 10 different files), agents can operate concurrently, reducing wall-clock time proportionally.

They lose on simple single-step tasks where the orchestration overhead (3-10x cost) outweighs any quality gain. They also struggle when tasks are deeply sequential with tight dependencies, since each handoff adds latency and potential for miscommunication. A good rule of thumb: if a single prompt with one tool call can solve it, do not build a multi-agent system.

## Why It Matters

1. **Specialization**: Individual agents can be optimized for narrow tasks with role-specific prompts, tools, and even different underlying models (e.g., a cheap fast model for drafting, a powerful expensive one for critical review).
2. **Scalability**: Complex workflows can be decomposed into parallelizable subtasks, enabling concurrent execution across agents and significantly reducing wall-clock time for large jobs.
3. **Error resilience**: Critic and reviewer agents catch mistakes that a single agent would miss, creating self-correcting systems with multiple layers of validation and quality assurance.
4. **Human-in-the-loop**: Multi-agent architectures naturally support human oversight at decision points -- a human can be modeled as just another agent in the conversation without disrupting the workflow.
5. **Emergent capability**: Agent teams can solve problems that no single agent can handle alone, particularly those requiring diverse tool use, multiple knowledge domains, and iterative refinement.

## Key Technical Details

- AutoGen group chat supports up to 10+ agents with automatic speaker selection via round-robin, random, or LLM-based routing
- CrewAI's hierarchical process uses a manager agent that makes 1 LLM call per delegation decision, adding ~20% overhead versus sequential
- LangGraph checkpointing enables time-travel debugging -- you can replay any state in the execution graph and branch from it
- Multi-agent systems typically incur 3-10x cost overhead versus single-agent due to inter-agent communication tokens
- Debate patterns work best when agents have access to different information sources or are prompted with different perspectives to avoid groupthink
- Latency scales with the number of sequential agent handoffs; parallel branches do not add latency if executed concurrently
- Shared memory stores (Redis, vector DBs) are common for blackboard communication patterns in production deployments
- Agent count sweet spot is typically 2-5 agents; beyond that, coordination overhead grows faster than quality gains
- Error propagation is the primary failure mode: one agent's bad output cascades through the pipeline unless explicit validation gates are added
- Token consumption in a 3-agent debate (3 rounds) is roughly 9x a single-agent call: 3 agents x 3 rounds, each reading the full conversation history
- Observability is critical: production multi-agent systems need logging of every agent call, token counts, latencies, and decision traces for debugging
- Cost can be optimized by using smaller/cheaper models for simple agent roles (summarizers, formatters) and reserving expensive models for complex reasoning agents

## Common Misconceptions

- **"More agents always means better results."** Adding agents increases communication overhead and can degrade performance on simple tasks. A single well-prompted agent often outperforms a multi-agent system for straightforward queries. The orchestration cost must be justified by the task complexity. Empirically, 2-5 agents is the sweet spot for most applications.
- **"Agents need to be different LLMs."** Most multi-agent systems use the same underlying model for all agents. Differentiation comes from system prompts, tools, and role definitions -- not different model weights. Using the same model with different personas is the standard approach. That said, using a cheaper model for simple subtasks and a more capable one for critical decisions is a valid cost-optimization strategy.
- **"Multi-agent debate always improves accuracy."** Debate helps on factual and reasoning tasks but can amplify errors when all agents share the same systematic biases or lack relevant knowledge. If every agent is wrong in the same way, debate just reinforces the error with higher confidence.
- **"Multi-agent systems are production-ready out of the box."** Most frameworks are still maturing. Production deployments require significant investment in observability, error handling, cost monitoring, and fallback strategies that the frameworks do not provide by default.

## Connections to Other Concepts

- **ReAct Pattern**: The foundational reasoning-action loop that most individual agents within a multi-agent system implement internally for their own tool use and reasoning.
- **Tool Use and Function Calling**: Agents' ability to invoke tools is what makes specialization practical -- different agents get different tool sets matched to their roles.
- **Memory Systems**: Shared and agent-specific memory enables context persistence across multi-turn agent interactions and long-running workflows spanning multiple sessions.
- **Self-Reflection**: Critic agents in multi-agent systems apply self-reflection principles to evaluate other agents' outputs, creating external feedback loops.
- **Reasoning Models**: Using reasoning models (o1, R1) as the backbone for critical agents (reviewers, planners) can improve multi-agent system quality on complex tasks.
- **RAG (Retrieval-Augmented Generation)**: Research agents in multi-agent systems typically use RAG to ground their outputs in retrieved documents before passing results to other agents.

## Further Reading

- Wu et al., "AutoGen: Enabling Next-Gen LLM Applications via Multi-Agent Conversation," arXiv:2308.08155, 2023
- Du et al., "Improving Factuality and Reasoning in Language Models through Multiagent Debate," 2023
- Hong et al., "MetaGPT: Meta Programming for A Multi-Agent Collaborative Framework," arXiv:2308.00352, 2023
- Li et al., "CAMEL: Communicative Agents for 'Mind' Exploration of Large Language Model Society," arXiv:2303.17760, 2023
- Qian et al., "ChatDev: Communicative Agents for Software Development," arXiv:2307.07924, 2023
- LangGraph documentation, "Multi-Agent Architectures," LangChain, 2024
