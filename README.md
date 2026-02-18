# LLM Concepts

A comprehensive reference of **153 concepts** covering the full landscape of Large Language Models -- from foundational transformer architecture to cutting-edge research in reasoning, alignment, and retrieval.

Each concept is a standalone deep dive written to be accessible yet technically rigorous. Every entry includes an intuitive analogy, detailed technical explanation, practical implications, common misconceptions, connections to related concepts, and references to key papers.

## Structure

The concepts are organized into 11 categories that follow the natural progression from architecture to application:

| # | Category | Concepts | Description |
|---|----------|----------|-------------|
| 01 | [Foundational Architecture](01-foundational-architecture/) | 20 | Transformer internals: attention, MoE, activation functions, autoregressive generation |
| 02 | [Input Representation](02-input-representation/) | 9 | Tokenization, positional encoding, embeddings, context windows |
| 03 | [Training Fundamentals](03-training-fundamentals/) | 16 | Pre-training, optimization, scaling laws, emergent abilities, data curation |
| 04 | [Distributed Training](04-distributed-training/) | 7 | Data/tensor/pipeline/expert parallelism, ZeRO, ring attention |
| 05 | [Alignment & Post-Training](05-alignment-and-post-training/) | 13 | RLHF, DPO, GRPO, Constitutional AI, reward modeling, synthetic data |
| 06 | [Parameter-Efficient Fine-Tuning](06-parameter-efficient-fine-tuning/) | 5 | LoRA, QLoRA, adapters, multi-LoRA serving |
| 07 | [Inference & Deployment](07-inference-and-deployment/) | 18 | KV cache, quantization, speculative decoding, continuous batching, vLLM |
| 08 | [Practical Applications](08-practical-applications/) | 11 | RAG, agents, tool use, prompt engineering, memory systems |
| 09 | [Safety & Alignment](09-safety-and-alignment/) | 21 | Jailbreaking, sleeper agents, hallucination, adversarial robustness, unlearning |
| 10 | [Evaluation](10-evaluation/) | 7 | Benchmarks, Chatbot Arena, contamination detection, LLM-as-judge |
| 11 | [Advanced & Emerging](11-advanced-and-emerging/) | 26 | Reasoning models, GraphRAG, mechanistic interpretability, state-space models |

## Concept Format

Every concept file follows a consistent structure:

```
# Concept Name

**One-Line Summary**: ...
**Prerequisites**: ...

## What Is [Concept]?
  (Intuitive analogy + accessible explanation)

## How It Works
  (Technical deep dive with subsections, code/pseudocode, diagrams)

## Why It Matters
  (Numbered list of practical implications)

## Key Technical Details
  (Bullet points with specific numbers, benchmarks, implementation notes)

## Common Misconceptions
  (Myth-busting with explanations)

## Connections to Other Concepts
  (Cross-references to related entries in this collection)

## Further Reading
  (Key papers and resources with descriptions)
```

## How to Use This Repository

**Sequential learning**: Start with Category 01 and work through in order. Each concept lists its prerequisites, so you can verify you have the background needed.

**Reference lookup**: Jump directly to any concept when you encounter it in a paper, blog post, or conversation. Each file is self-contained.

**Prerequisite tracing**: Use the "Prerequisites" field and "Connections" section to build a learning path for any topic.

**Interview/study prep**: The "Common Misconceptions" sections highlight the nuances that distinguish deep understanding from surface knowledge.

## Quick Start

Some recommended starting points by interest:

- **"How do LLMs actually work?"** -- Start with [Self-Attention](01-foundational-architecture/self-attention.md) then [Transformer Architecture](01-foundational-architecture/transformer-architecture.md)
- **"How are models trained?"** -- [Pre-Training](03-training-fundamentals/pre-training.md) then [Scaling Laws](03-training-fundamentals/scaling-laws.md)
- **"How is ChatGPT aligned?"** -- [RLHF](05-alignment-and-post-training/rlhf.md) then [DPO](05-alignment-and-post-training/dpo.md)
- **"How do I build with LLMs?"** -- [RAG](08-practical-applications/rag.md) then [AI Agents](08-practical-applications/ai-agents.md)
- **"What are the latest advances?"** -- [Reasoning Models](11-advanced-and-emerging/reasoning-models.md) then [Inference-Time Scaling Laws](11-advanced-and-emerging/inference-time-scaling-laws.md)
- **"What are the safety concerns?"** -- [Sleeper Agents](09-safety-and-alignment/sleeper-agents.md) then [Adversarial Robustness](09-safety-and-alignment/adversarial-robustness.md)

## Full Concept Index

See [OUTLINE.md](OUTLINE.md) for the complete list of all 153 concepts with one-line summaries.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on adding new concepts or improving existing ones.

## License

This project is licensed under the MIT License -- see [LICENSE](LICENSE) for details.
