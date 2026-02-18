# Structured Output & JSON Mode

**One-Line Summary**: Structured output techniques constrain LLM generation to produce reliably parseable formats like JSON, XML, or YAML, transforming probabilistic text generation into deterministic, schema-conformant outputs essential for software integration.

**Prerequisites**: Understanding of how LLM token generation works (autoregressive sampling from a vocabulary), familiarity with JSON and JSON Schema, and basic knowledge of how LLMs are used in software systems.

## What Is Structured Output?

Imagine asking someone to fill out a tax form. You do not want them to write a free-form essay about their finances -- you need specific numbers in specific boxes in a specific format. Structured output is the equivalent for LLMs: instead of letting the model produce arbitrary text, you constrain it to produce output that conforms to a predefined structure.

LLMs naturally produce free-form text. But software systems need structured data -- JSON objects, XML documents, database entries, API payloads. Structured output bridges this gap by ensuring the model's output is not just semantically correct but *syntactically valid* and *schema-compliant*. This means downstream code can parse the output with confidence, without fragile regex extraction or error-prone string manipulation.

## How It Works

### Approach 1: Prompt-Based Instruction

The simplest approach is asking the model to output a specific format in the prompt:

```
Extract the following information as JSON:
{"name": string, "age": number, "occupation": string}

Text: "Dr. Sarah Chen, 42, is a neurosurgeon at Mass General."
```

This works surprisingly often with capable models, but it provides no guarantees. The model might add markdown code fences around the JSON, include a preamble ("Here's the extracted information:"), produce invalid JSON (trailing commas, unescaped quotes), or deviate from the schema (extra fields, wrong types). For production systems, prompt-based approaches alone are insufficient.

### Approach 2: Constrained Decoding (Grammar-Based Token Filtering)

This is the technically elegant solution. At each step of token generation, the model produces logits (scores) for every token in its vocabulary. Normally, any token can be selected. Constrained decoding intervenes by masking out tokens that would lead to invalid output.

For example, if you are generating JSON and the model has just produced `{"name": "Sarah`, the only valid next tokens are characters that continue the string or a closing `"`. Tokens like `{`, `[`, or digits are masked (their logits set to negative infinity), making them impossible to select.

This is implemented by compiling the target format (JSON Schema, regular expression, context-free grammar) into a finite-state machine or pushdown automaton. At each generation step, the automaton determines which tokens are valid continuations, and only those tokens participate in sampling.

**Key libraries**:

- **Outlines**: An open-source library that compiles JSON schemas and regex patterns into efficient token-level masks. Works with any model accessible via its logits (Hugging Face transformers, vLLM, etc.).
- **Guidance** (from Microsoft): Provides a template language where you interleave fixed text with model-generated segments, each potentially constrained by a grammar or regex.
- **llama.cpp grammars**: Supports GBNF (a variant of BNF grammar notation) for constraining generation at the C++ level, enabling structured output even in quantized local models.
- **SGLang**: Includes constrained decoding as part of its runtime for efficient structured generation.

### Approach 3: Provider-Level JSON Mode

Major API providers have built structured output into their services:

**OpenAI's Structured Outputs**: Accepts a JSON Schema alongside the prompt and guarantees the response conforms to it. Internally uses constrained decoding. Supports complex schemas including nested objects, arrays, enums, and optional fields.

**Anthropic's tool use**: Returns structured JSON as part of tool call responses, with schema validation against the provided tool definitions.

**vLLM's JSON mode**: The open-source inference engine supports guided decoding using Outlines or lm-format-enforcer, enabling structured output for any model served through it.

**Google Gemini's response schema**: Accepts a schema definition and constrains output accordingly.

### Approach 4: Post-Processing with Validation and Retry

A pragmatic fallback: generate unconstrained output, attempt to parse it, and if parsing fails, retry with an error message appended to the prompt. This is less elegant but works with any model, even through APIs that do not support constrained decoding.

A typical retry loop:
1. Generate response with format instructions
2. Attempt to parse as JSON
3. If parsing fails, send a follow-up: "Your previous response was not valid JSON. The error was: {error}. Please try again, outputting only valid JSON."
4. Repeat up to N times

Libraries like **Instructor** (Python) and **Zod** (TypeScript, via the Vercel AI SDK) automate this pattern with schema validation, automatic retries, and type-safe output.

## Common Output Formats

**JSON**: The dominant format for structured LLM output. Nearly universal API support, excellent tooling, and strong model familiarity from training data. The clear default choice.

**XML**: Less common but useful when output needs to represent hierarchical document structures or when integrating with XML-based systems. Some models handle XML well due to extensive XML in training data.

**YAML**: Occasionally used for configuration-style outputs. More human-readable than JSON but harder to constrain precisely due to its flexible syntax.

**Markdown with structure**: Semi-structured output using headers, tables, and lists. Not formally parseable like JSON but useful for human-facing applications that need some consistency.

**CSV/TSV**: For tabular data. Simple but limited to flat structures.

## Why It Matters

Structured output is the linchpin of LLM integration into software systems. Without it, every LLM output requires fragile parsing code, every edge case is a potential production failure, and reliability remains unpredictable. With it, LLM outputs become as reliable as any other API response -- typed, validated, and guaranteed to parse.

This capability enables LLMs to serve as components in larger software architectures rather than standalone chatbots. When an LLM reliably produces `{"sentiment": "positive", "confidence": 0.92, "topics": ["pricing", "support"]}`, it can be a drop-in replacement for a classification service, feeding directly into dashboards, databases, and downstream logic.

## Key Technical Details

- **Constrained decoding can affect quality**: Forcing the model down a narrow token path may prevent it from expressing nuance or reaching the best answer if the grammar is too restrictive. Well-designed schemas mitigate this.
- **Schema complexity limits**: Very deeply nested schemas or schemas with many required fields can slow constrained decoding and occasionally degrade output quality. Keep schemas as simple as the task allows.
- **Token overhead**: Structured formats add token overhead. JSON keys, braces, and quotes consume tokens that could otherwise be used for content. This is generally a small cost but matters in token-constrained settings.
- **Streaming compatibility**: Constrained decoding is compatible with streaming -- each token is still generated sequentially, just from a filtered set. Partial JSON can be streamed and progressively parsed.
- **Null handling**: Define explicitly whether fields can be null. Models sometimes produce `null` for fields they are uncertain about, which is usually preferable to hallucinating a value.

## Common Misconceptions

**"JSON mode means the content is correct."** Structured output guarantees *format*, not *factual accuracy*. A model can produce perfectly valid JSON with completely hallucinated values. Schema conformance and content quality are orthogonal.

**"Constrained decoding is slow."** Modern implementations (Outlines, vLLM) compile grammars ahead of time and add minimal per-token overhead. For most schemas, the latency impact is negligible.

**"You need structured output for everything."** Free-form text is the right output for many tasks -- creative writing, summarization, conversational responses. Structured output is specifically for when the output feeds into software systems that need reliable parsing.

**"Retry loops are unreliable."** With good models and clear error messages, retry loops succeed on the second attempt in 95%+ of cases. Combined with validation libraries, they are a robust production pattern, especially when constrained decoding is not available.

## Connections to Other Concepts

- **Function calling** is the primary consumer of structured output -- tool call arguments must be valid JSON conforming to tool schemas.
- **AI agents** depend on structured output to communicate actions, parse observations, and maintain state.
- **Prompt engineering** includes output format specification as a core technique.
- **Tokenization** interacts with constrained decoding: the token vocabulary determines what atomic units the grammar must reason about.
- **Inference optimization** intersects with constrained decoding -- speculative decoding and batched generation must be compatible with per-token grammar constraints.

## Further Reading

- Willard, B. & Louf, R. (2023). "Efficient Guided Generation for Large Language Models." The foundational paper behind the Outlines library, describing how to compile regular expressions and context-free grammars into token-level masks.
- OpenAI (2024). "Introducing Structured Outputs in the API." Documentation describing OpenAI's approach to guaranteed schema conformance.
- Zheng, L. et al. (2024). "SGLang: Efficient Execution of Structured Language Model Programs." Describes a runtime that co-optimizes structured generation with batching and caching for high throughput.
