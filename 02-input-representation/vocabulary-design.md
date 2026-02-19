# Vocabulary Design

**One-Line Summary**: Vocabulary design is the process of choosing how many and which tokens a language model should know, balancing compression efficiency against embedding size, multilingual coverage, and tokenization fairness across languages.

**Prerequisites**: Understanding of tokenization and BPE (how vocabularies are constructed), token embeddings (how vocabulary size affects model parameters), and awareness of multilingual NLP challenges.

## What Is Vocabulary Design?

Imagine you're designing an alphabet for a new universal language. Too few characters and every word requires long, cumbersome sequences of symbols. Too many characters and people must memorize thousands of glyphs, many of which they'll rarely use. The sweet spot depends on who will use the language, what they'll write about, and how much memory they have for learning symbols.

![Chart showing tokenization fertility (tokens per word) across different languages, illustrating the disparity between English and non-Latin-script languages](https://raw.githubusercontent.com/openai/tiktoken/main/scripts/vocab_size_comparison.png)
*Source: [OpenAI tiktoken – Vocabulary Size Comparison](https://github.com/openai/tiktoken)*


Vocabulary design for LLMs faces exactly this trade-off. The vocabulary is the fixed set of tokens (subword units) that the model can recognize and produce. Every piece of text the model reads or writes must be expressed using only these tokens. The vocabulary is typically finalized before model training begins and remains immutable throughout the model's lifetime.

Choices about vocabulary size, the training corpus used to build it, and whether to use byte-level or character-level base units have profound downstream effects on model performance, efficiency, and fairness.

## How It Works


*See diagram comparing vocabulary allocation strategies at: [Petrov et al., "Language Model Tokenizers Introduce Unfairness Between Languages" (NeurIPS 2024)](https://arxiv.org/abs/2305.15425) -- includes figures showing token fertility ratios across 350+ languages and the impact of vocabulary size on multilingual equity.*

### Choosing Vocabulary Size

Vocabulary sizes in modern LLMs typically fall between 32,000 and 256,000 tokens:

| Model | Vocabulary Size |
|-------|----------------|
| LLaMA 1 | 32,000 |
| GPT-2 | 50,257 |
| LLaMA 2/3 | 32,000 / 128,256 |
| Mistral | 32,000 |
| GPT-4 (cl100k) | 100,277 |
| GPT-4o (o200k) | 200,019 |
| Gemma | 256,000 |
| DeepSeek-V2 | 100,015 |

The choice is governed by a core trade-off.

**Larger vocabulary benefits**:
- Better compression: common words and phrases get single tokens, meaning fewer tokens per text, which extends effective context and reduces per-token costs.
- Faster inference: fewer tokens to generate means faster response times.
- Dedicated tokens for common multi-character sequences (e.g., `" the"`, `"function"`, `"return"` in code).

**Larger vocabulary costs**:
- Larger embedding matrix: $|V| \times d$ parameters. For $|V| = 256{,}000$ and $d = 4{,}096$, the embedding matrix alone is ~1 billion parameters (plus the same for the output projection if not weight-tied).
- Sparser training signal: each token appears less frequently in the training data, making it harder to learn good embeddings for rare tokens.
- Diminishing returns: the 200,000th token in the vocabulary captures a much rarer pattern than the 10,000th.

A rough guideline from scaling laws research: vocabulary size should scale as approximately:

$$|V| \propto N^{0.5}$$

where $N$ is the total parameter count of the model, though practical choices diverge from this.

### The Compression vs. Fertility Trade-off

**Compression ratio** measures how efficiently a tokenizer converts text to tokens (characters per token). Higher is better.

**Fertility** measures how many tokens a single word produces on average. Lower is better. A fertility of 1.0 means every word is a single token; 3.0 means each word becomes 3 tokens.

For English, modern tokenizers achieve fertility around 1.2-1.4 (very efficient). For other languages, fertility can be dramatically worse:

| Language | Approximate Fertility (GPT-4) |
|----------|------------------------------|
| English | 1.3 |
| Spanish | 1.5 |
| Chinese | 1.8 |
| Hindi | 3.0 |
| Burmese | 6.0+ |
| Amharic | 8.0+ |

This disparity means a Hindi speaker's prompt uses ~2.3x more tokens than an equivalent English prompt, effectively giving them a smaller context window and higher costs for the same amount of meaning.

### Multilingual Vocabulary Challenges

Building a vocabulary for a multilingual model presents a fundamental allocation problem: the vocabulary has a fixed budget, and every token dedicated to Mandarin is one fewer token for English, and vice versa.

**Strategies include**:

1. **Proportional allocation**: Dedicate vocabulary budget proportionally to the amount of each language in the training data. This tends to overserve English and underserve low-resource languages.

2. **Temperature-based sampling**: Modify the language distribution used for tokenizer training with a temperature parameter $\alpha$:

$$p_l' = \frac{p_l^\alpha}{\sum_k p_k^\alpha}$$

where $p_l$ is the natural proportion of language $l$ and $\alpha < 1$ flattens the distribution, giving more vocabulary budget to underrepresented languages. Common values are $\alpha = 0.3$ to $\alpha = 0.7$.

3. **Script-aware allocation**: Ensure each writing system has adequate base coverage before allocating subword merges.

4. **Separate tokenizers per language group**: Train different tokenizers for different language families. This is impractical for a single model but used in some specialized systems.

### Byte-Level vs. Character-Level vs. Subword

**Character-level base**: The initial vocabulary consists of all unique characters in the training data. This can be tens of thousands for multilingual corpora (Unicode has ~150,000 assigned characters). Rare characters become `<UNK>`.

**Byte-level base**: The initial vocabulary consists of exactly 256 byte values. Every possible byte sequence can be represented, guaranteeing no unknown tokens. GPT-2 introduced this for BPE and it's now standard. The cost: non-ASCII characters require multiple base tokens before merges compress them.

**Byte-level models (no subword tokenization)**: Models like ByT5 and MambaByte operate directly on bytes with no learned vocabulary at all. This eliminates the tokenizer entirely but produces sequences 3-5x longer than subword models, requiring architectures that handle long sequences efficiently. This approach has gained interest as a potential future direction because it:
- Eliminates all tokenization artifacts and biases
- Is inherently language-agnostic
- Can handle any input (code, binary data, novel scripts)
- Removes the fixed-vocabulary constraint

However, byte-level models currently lag behind subword models in efficiency and performance at scale, though the gap is narrowing.

### Vocabulary Composition

Beyond size, the composition of the vocabulary matters. A well-designed vocabulary includes:

*See also the vocabulary composition analysis at: [Hugging Face Blog – Tokenizers: How Machines Read](https://huggingface.co/docs/tokenizers/) -- includes visualizations of how different vocabulary sizes affect compression ratios and the trade-off between vocabulary coverage and embedding matrix size.*


- **Base tokens**: Individual bytes or characters (256 for byte-level).
- **Common words**: High-frequency words in the training languages.
- **Morphological units**: Prefixes, suffixes, and stems that enable compositional understanding.
- **Whitespace-prefixed variants**: `" the"` (with a leading space) is typically a different token from `"the"`. This encodes word boundary information.
- **Code tokens**: Indentation (`"    "`, `"\t"`), common programming keywords, operators.
- **Numeric tokens**: Individual digits, common numbers, or multi-digit sequences.
- **Special tokens**: BOS, EOS, PAD, and any task-specific control tokens.

## Why It Matters

Vocabulary design decisions are among the most permanent choices in LLM development. Unlike model architecture or training hyperparameters, the vocabulary cannot be changed after training without retraining the entire model. This makes it a high-stakes decision with lasting consequences:

- **Multilingual fairness**: A vocabulary optimized for English systematically disadvantages non-English users in cost, context utilization, and quality.
- **Domain performance**: A vocabulary that doesn't include common code tokens will tokenize Python inefficiently, wasting capacity in code-focused applications.
- **Model economics**: Vocabulary size affects the embedding matrix, which can be 5-30% of total parameters in smaller models. For a 7B model, the difference between 32K and 128K vocabulary is ~400M additional parameters.
- **Downstream behavior**: Tokenization fertility affects everything from chain-of-thought reasoning (more tokens per thought step) to classification (how class labels are tokenized) to arithmetic (whether digits are individual tokens).

## Key Technical Details

- Adding new tokens to a pretrained model is possible (embedding resizing) but the new token embeddings must be trained, typically through fine-tuning. The original model had no training signal for these tokens.
- Vocabulary overlaps between models are often low. GPT-4's and LLaMA's vocabularies share only a fraction of their tokens because they were trained on different corpora with different BPE merge sequences.
- Token frequency follows a Zipfian distribution: a small number of tokens (a few hundred) account for the vast majority of all text, while the long tail of rare tokens each appear infrequently.
- The output softmax layer computes $\text{softmax}(\mathbf{h} \mathbf{E}^T)$, which is a dot product of the hidden state with every vocabulary token's embedding. Larger vocabularies make this computation more expensive, though optimizations like adaptive softmax and vocabulary parallelism mitigate this.
- Vocabulary design interacts with pre-tokenization rules. Splitting digits individually (as GPT-4 does) versus keeping multi-digit numbers together dramatically affects arithmetic ability.

## Common Misconceptions

- **"Bigger vocabulary is always better."** There are real costs: more parameters, sparser training signal, higher softmax computation cost. The optimal vocabulary size depends on model size, training data, and target languages.
- **"Vocabulary size doesn't matter much."** It matters enormously for non-English languages, code, and domain-specific applications. Moving from 32K to 128K vocabulary improved LLaMA 3's multilingual and code performance substantially.
- **"The vocabulary is just a list of words."** Most vocabulary entries are subwords, not words. The vocabulary includes partial words, punctuation sequences, whitespace patterns, and byte sequences that don't correspond to readable text.
- **"All languages benefit equally from a larger vocabulary."** Languages with Latin scripts benefit disproportionately from English-centric training corpora. Increasing vocabulary from 32K to 128K may add thousands of English subwords but only dozens of Bengali subwords, depending on the training corpus.

## Connections to Other Concepts

- **Tokenization / BPE**: The tokenization algorithm builds the vocabulary. Vocabulary size is the primary hyperparameter controlling tokenizer training.
- **Token Embeddings**: Vocabulary size directly determines the shape of the embedding matrix ($|V| \times d$).
- **Context Window**: Tokenization fertility (determined by vocabulary) controls how much text fits in the fixed-token-length context window.
- **Special Tokens**: These are manually added to the vocabulary outside the normal BPE process.
- **Byte-Pair Encoding**: The specific algorithm most commonly used to construct the vocabulary from a training corpus.

## Further Reading

- Rust, P., et al. (2021). "How Good is Your Tokenizer? On the Monolingual Performance of Multilingual Language Models." *ACL 2021.* -- Rigorous analysis of how tokenizer quality varies across languages and its impact on model performance.
- Liang, D., et al. (2023). "XLM-V: Overcoming the Vocabulary Bottleneck in Multilingual Masked Language Models." *EMNLP 2023.* -- Proposes methods for improving multilingual vocabulary allocation.
- Clark, J.H., Garrette, D., Turc, I., & Wieting, J. (2022). "Canine: Pre-training an Efficient Tokenization-Free Encoder for Language Representation." *TACL.* -- Explores character-level and tokenization-free approaches as an alternative to subword vocabularies.
