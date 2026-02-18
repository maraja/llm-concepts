# Contributing

Thank you for your interest in improving this collection. Whether you're fixing a factual error, improving an explanation, or adding a new concept, contributions are welcome.

## Concept Format

Every concept file must follow this structure:

```markdown
# Concept Name

**One-Line Summary**: A single sentence capturing the essence.

**Prerequisites**: Comma-separated list of concepts the reader should understand first.

## What Is [Concept]?

Start with an intuitive analogy that a non-specialist could follow,
then bridge to the technical definition.

## How It Works

Technical deep dive with subsections as needed.
Include code snippets, pseudocode, or diagrams where they help.

## Why It Matters

Numbered list of practical implications and real-world impact.

## Key Technical Details

- Bullet points with specific numbers, benchmarks, and implementation notes
- Prefer concrete details over vague claims

## Common Misconceptions

- **"Misconception statement."** Explanation of why it's wrong and what's actually true.

## Connections to Other Concepts

- **Related Concept**: How this concept connects to others in the collection.

## Further Reading

- Author et al., "Paper Title" (Year) -- Brief description of why this paper matters.
```

## Guidelines

### Adding a New Concept

1. Check OUTLINE.md to verify the concept isn't already covered (possibly under a different name).
2. Choose the most appropriate category directory (01-11).
3. Name the file using lowercase-kebab-case: `my-new-concept.md`.
4. Follow the format template above.
5. Aim for 100-200 lines -- thorough but not exhaustive.
6. Include at least one intuitive analogy in the "What Is" section.
7. Cross-reference at least 2-3 related concepts from the existing collection.
8. Cite primary papers with author, title, and year.

### Improving an Existing Concept

- **Factual corrections**: Cite a source when correcting claims.
- **Clarity improvements**: If something is confusing, simplify it. Don't add complexity.
- **Missing information**: Add what's missing without inflating the word count unnecessarily.
- **Updated benchmarks**: As the field moves fast, newer numbers are welcome with citations.

### What to Avoid

- Marketing language or hype ("revolutionary", "groundbreaking").
- Unsourced claims about model performance.
- Excessive length -- if a section grows beyond the template, consider whether a new concept file is more appropriate.
- Duplicate coverage -- check if the topic is already addressed in another file.

## File Placement

| Category | What Goes Here |
|----------|---------------|
| `01-foundational-architecture/` | Core transformer components and architectural variants |
| `02-input-representation/` | Tokenization, positional encoding, embeddings |
| `03-training-fundamentals/` | Optimization, loss functions, scaling, data |
| `04-distributed-training/` | Parallelism strategies and distributed systems |
| `05-alignment-and-post-training/` | RLHF, DPO, reward modeling, preference learning |
| `06-parameter-efficient-fine-tuning/` | LoRA, adapters, and related methods |
| `07-inference-and-deployment/` | Serving, decoding, caching, quantization |
| `08-practical-applications/` | RAG, agents, tool use, prompt engineering |
| `09-safety-and-alignment/` | Attacks, defenses, alignment failures, guardrails |
| `10-evaluation/` | Benchmarks, metrics, evaluation methodology |
| `11-advanced-and-emerging/` | Cutting-edge research and emerging techniques |

## Pull Request Process

1. Fork the repository.
2. Create a branch: `git checkout -b add-concept-name` or `git checkout -b fix-concept-name`.
3. Make your changes following the guidelines above.
4. Submit a pull request with a brief description of what you changed and why.
