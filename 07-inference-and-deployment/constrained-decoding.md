# Constrained Decoding

**One-Line Summary**: Constrained decoding forces LLM output to conform to formal grammars (JSON schemas, regex patterns, context-free grammars) by masking invalid tokens at each decoding step, providing a 100% structural validity guarantee and eliminating retry loops for malformed output.

**Prerequisites**: Autoregressive generation, logits and softmax, tokenization (BPE), sampling strategies, JSON/regex/formal grammar basics.

## What Is Constrained Decoding?

Imagine a writer composing a sonnet. Normally they have complete freedom -- any word, any structure. Now imagine an editor sitting beside them who, after each word, instantly crosses out every possible next word that would violate the sonnet's rules: wrong syllable count, broken rhyme scheme, incorrect meter. The writer still chooses freely among the remaining valid words, preserving their creative style, but the structural constraints are guaranteed to be met. The editor never changes the writer's preferences -- they only remove options that would break the form.

![Outlines structured generation overview showing how a finite-state machine guides token generation to produce valid JSON](https://raw.githubusercontent.com/dottxt-ai/outlines/main/docs/assets/images/logo.png)
*See diagram at: [Outlines - Structured Generation Library](https://github.com/dottxt-ai/outlines)*


This is exactly what constrained decoding does to an LLM. At each generation step, the model produces logits (scores) for every token in its vocabulary. Before sampling, a constraint engine masks tokens that would produce structurally invalid output -- setting their logits to negative infinity so they have zero probability of being selected. The model then samples from the remaining valid tokens using its normal probability distribution.

The power of this approach is its guarantee: the output is 100% structurally valid by construction, not by hope. Without constrained decoding, developers resort to fragile retry loops -- generate output, parse it, if parsing fails retry with a scolding prompt -- which wastes compute, adds latency, and still does not guarantee success. Constrained decoding eliminates this failure mode entirely.

## How It Works


![XGrammar constrained decoding architecture showing grammar compilation and token masking pipeline](https://raw.githubusercontent.com/mlc-ai/blog/main/img/xgrammar/xgrammar-overview.svg)
*See diagram at: [XGrammar: Flexible and Efficient Structured Generation (MLC Blog)](https://blog.mlc.ai/2024/11/22/achieving-efficient-flexible-portable-structured-generation-with-xgrammar)*

### Grammar Compilation to Automata

The constraint specification (JSON schema, regex, CFG) is compiled into a finite automaton or pushdown automaton before inference begins. This automaton tracks the "state" of the output so far and, at each state, knows exactly which characters are valid next.

```
JSON object grammar (simplified):
  Start → '{' → key_string → ':' → value → (',' → key_string → ':' → value)* → '}'

For regex: r'"name"\s*:\s*"[A-Za-z ]+"'
  Compiled to a DFA with states tracking position in the pattern.
```

At each decoding step, the system:
1. Determines the current automaton state based on all tokens generated so far.
2. Computes the set of valid next characters from this state.
3. Maps valid characters back to valid vocabulary tokens.
4. Sets logits for all invalid tokens to negative infinity.
5. Lets the model sample normally from the remaining valid tokens.

### The Token-Boundary Challenge

This is the core engineering challenge of constrained decoding. LLMs use BPE (Byte Pair Encoding) tokenization, where a single token can represent multiple characters (e.g., the token `" name"` is 5 characters including the leading space). The constraint engine must reason at the character level, but masking operates at the token level.

For each vocabulary token, the system must determine: "If I append this token's characters to the output so far, does the resulting string still have at least one valid completion according to the grammar?"

```python
# Pseudocode for token-level mask computation
def compute_valid_tokens(automaton_state, vocabulary):
    valid_tokens = []
    for token in vocabulary:
        # Simulate appending this token's characters
        chars = token.decode()
        state = automaton_state
        valid = True
        for char in chars:
            state = automaton.transition(state, char)
            if state is None:  # No valid transition
                valid = False
                break
        if valid:
            # Check that some completion exists from this state
            if automaton.has_valid_continuation(state):
                valid_tokens.append(token)
    return valid_tokens
```

This per-token analysis of the entire vocabulary (typically 32K-128K tokens) at every decoding step could be prohibitively expensive. Modern implementations use several optimizations.

### Efficient Implementations

**XGrammar** (MLC/TVM team) achieves less than 1 millisecond overhead per decoding step through:
- **Precomputation**: For each automaton state, the set of valid tokens is precomputed and cached. Since grammars have finite states, this cache is bounded.
- **Adaptive token masking**: Tokens are partitioned into "context-independent" tokens (always valid or always invalid regardless of boundary alignment) and "context-dependent" tokens (validity depends on the exact character boundary). Only the latter require per-step analysis.
- **Pushdown automaton support**: Handles recursive grammars (nested JSON, nested parentheses) that finite automata cannot express.

**Outlines** (dottxt-ai) compiles regex patterns and JSON schemas into finite-state machines and precomputes token masks per state:
```python
import outlines

model = outlines.models.transformers("mistralai/Mistral-7B-v0.1")

# JSON schema constraint
schema = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "age": {"type": "integer", "minimum": 0},
        "city": {"type": "string"}
    },
    "required": ["name", "age"]
}
generator = outlines.generate.json(model, schema)
result = generator("Generate a person's info:")
# result is GUARANTEED to be valid JSON matching the schema
```

**Guidance** (Microsoft) takes a template-based approach where the user interleaves fixed text with generation blocks, each optionally constrained:
```python
from guidance import models, gen, select

lm = models.Transformers("gpt2")
lm += "Answer: {" + '"sentiment": "' + select(["positive", "negative", "neutral"]) + '"}'
```

### Production Deployment

OpenAI's "Structured Outputs" and Anthropic's tool use with JSON schemas implement constrained decoding server-side. When you specify a JSON schema in the API:

*See diagram of token masking during constrained decoding at: [Guidance - Microsoft (GitHub)](https://github.com/guidance-ai/guidance)*


```python
response = client.chat.completions.create(
    model="gpt-4o",
    response_format={
        "type": "json_schema",
        "json_schema": {
            "name": "person",
            "schema": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "age": {"type": "integer"}
                },
                "required": ["name", "age"]
            }
        }
    },
    messages=[{"role": "user", "content": "Generate a person"}]
)
# response.choices[0].message.content is guaranteed valid JSON
```

The server applies token masking internally, so the API consumer receives a 100% validity guarantee without needing to implement any parsing or retry logic.

## Why It Matters

1. **Eliminates structural failures**: In production systems that parse LLM output (API backends, data pipelines, agent tool calls), malformed output causes cascading failures. Constrained decoding provides a compile-time-like guarantee at generation time.
2. **Removes retry loops**: Without constrained decoding, developers implement retry loops that waste 2-5x compute on average and still have a non-zero failure rate. Constrained decoding succeeds on the first attempt, every time.
3. **Enables reliable tool use**: LLM agents that call tools via structured JSON must produce valid function calls. Constrained decoding is the foundation of reliable agentic workflows.
4. **Preserves model quality**: Unlike post-hoc parsing or string manipulation, constrained decoding lets the model use its full distribution over valid tokens. The model's semantic preferences are preserved; only structurally invalid options are removed.
5. **Simplifies application code**: Downstream code can assume valid structure, eliminating defensive parsing, error handling for malformed output, and complex validation logic.

## Key Technical Details

- **Overhead**: XGrammar achieves <1ms overhead per token. Outlines and Guidance add 1-5ms depending on grammar complexity and vocabulary size. This is typically negligible compared to the model's own inference time (10-50ms per token).
- **Grammar expressiveness**: Regex and JSON schemas cover most practical use cases. Full CFGs (context-free grammars) handle recursive structures. Some tools also support EBNF (Extended Backus-Naur Form) for maximum flexibility.
- **Interaction with sampling**: Constrained decoding modifies logits before sampling. It works with any sampling strategy (greedy, top-k, top-p, temperature scaling) -- the constraint mask is applied to the raw logits first, then normal sampling proceeds.
- **Beam search compatibility**: Constrained decoding composes naturally with beam search. Each beam independently maintains its automaton state and token mask.
- **Vocabulary size impact**: Larger vocabularies (128K tokens) make per-step mask computation more expensive. Precomputation and caching are essential at this scale.
- **Quality consideration**: While structural validity is guaranteed, semantic quality is not. The model might produce valid JSON with nonsensical values. Constrained decoding solves the *form* problem, not the *content* problem.
- **Framework support**: vLLM and TensorRT-LLM both offer native grammar-based decoding. SGLang integrates XGrammar for high-performance constrained generation.

## Common Misconceptions

- **"Constrained decoding hurts output quality."** When the grammar is well-designed, the constraint only removes tokens that would produce invalid output -- tokens the model should not choose anyway. Quality degradation is minimal because the model's probability mass over structurally valid continuations is typically high. Poorly designed grammars that over-constrain can reduce quality, but that is a grammar design issue, not a constrained decoding issue.
- **"You can achieve the same thing with prompt engineering."** Prompting the model to "output valid JSON" reduces but does not eliminate structural errors. At scale (millions of requests), even a 1% failure rate means thousands of failures per day. Constrained decoding achieves exactly 0% structural failure rate.
- **"Constrained decoding is too slow for production."** Modern implementations (XGrammar) add less than 1ms per token. For a model generating tokens at 30-50ms each, this is under 3% overhead.
- **"Regular expressions are sufficient for all constraints."** Regex cannot express recursive structures (nested JSON objects, balanced parentheses). CFGs or pushdown automata are needed for these cases. Most practical JSON schemas require CFG-level expressiveness.

## Connections to Other Concepts

- **Logits and Softmax**: Constrained decoding operates directly on logits, setting invalid tokens to negative infinity before softmax converts them to probabilities. Understanding the logit-to-probability pipeline is essential.
- **Sampling Strategies**: Constrained decoding composes with all sampling strategies. The mask is applied first, then temperature, top-k, top-p, etc. operate on the reduced valid set.
- **Tokenization (BPE)**: The token-boundary problem is a direct consequence of BPE's variable-length token-to-character mapping. Understanding BPE is essential for grasping why constrained decoding is non-trivial.
- **Model Serving Frameworks**: vLLM, TGI, and TensorRT-LLM all integrate constrained decoding, making it available as a serving-layer feature rather than requiring client-side implementation.
- **Speculative Decoding**: Constrained decoding interacts with speculative decoding -- draft tokens must also be checked against the grammar, and rejected tokens may need grammar state rollback.

## Further Reading

- Willard and Louf, "Efficient Guided Generation for Large Language Models" (2023) -- The Outlines paper, introducing efficient finite-state machine compilation for regex and JSON schema constraints.
- Dong et al., "XGrammar: Flexible and Efficient Structured Generation Engine for Large Language Models" (2024) -- Achieves sub-millisecond overhead through adaptive token masking and precomputation.
- Zheng et al., "SGLang: Efficient Execution of Structured Language Model Programs" (2024) -- Integrates constrained decoding with the broader LLM programming framework.
