# Byte-Pair Encoding (BPE)

**One-Line Summary**: Byte-Pair Encoding is a data compression algorithm repurposed for tokenization that iteratively merges the most frequent pair of adjacent symbols to build a subword vocabulary from the bottom up.

**Prerequisites**: Understanding of what tokenization is and why subword approaches are preferred over character-level or word-level tokenization. Basic familiarity with frequency counting and greedy algorithms.

## What Is Byte-Pair Encoding?

Imagine you're inventing a shorthand system for writing. You start by looking at your notes and noticing that "th" appears together constantly. So you invent a single symbol for "th." Then you notice "the" appears often (now representable as your new "th" symbol plus "e"), so you merge those. You keep going, always combining the most frequent pair, until you have a shorthand vocabulary of a desired size.

That's BPE in a nutshell. Originally developed by Philip Gage in 1994 as a data compression technique, it was adapted for neural machine translation by Sennrich et al. in 2016 and has since become the dominant tokenization algorithm in modern LLMs. GPT-2, GPT-3, GPT-4, LLaMA, Mistral, and most leading models use BPE or close variants.

## How It Works

### Training the Tokenizer (Building the Vocabulary)

BPE training is a straightforward iterative process. Let's walk through a concrete example.

**Step 0: Start with a corpus and initialize the vocabulary with individual characters.**

Suppose our training corpus consists of these words with their frequencies:

```
"low"    : 5 times   -> ['l', 'o', 'w']
"lower"  : 2 times   -> ['l', 'o', 'w', 'e', 'r']
"newest" : 6 times   -> ['n', 'e', 'w', 'e', 's', 't']
"widest" : 3 times   -> ['w', 'i', 'd', 'e', 's', 't']
```

Initial vocabulary: `{l, o, w, e, r, n, s, t, i, d}`

**Step 1: Count all adjacent pairs and find the most frequent.**

| Pair    | Count |
|---------|-------|
| (l, o)  | 7 (5 from "low" + 2 from "lower") |
| (o, w)  | 7 |
| (w, e)  | 8 (2 from "lower" + 6 from "newest") |
| (e, r)  | 2 |
| (n, e)  | 6 |
| (e, s)  | 9 (6 from "newest" + 3 from "widest") |
| (s, t)  | 9 |
| (w, i)  | 3 |
| (i, d)  | 3 |
| (d, e)  | 3 |

Tie between (e, s) and (s, t) at 9. Suppose we pick `(e, s)` first (ties are broken deterministically by implementation).

**Merge**: Create new token `es`. Update all sequences:

```
"newest" : ['n', 'e', 'w', 'es', 't']
"widest" : ['w', 'i', 'd', 'es', 't']
```

Add `es` to vocabulary. Record merge rule: `e + s -> es`.

**Step 2: Recount pairs, merge the next most frequent.**

Now (es, t) appears 9 times. Merge to form `est`.

**Step 3: Continue** until the vocabulary reaches the target size (e.g., 32,000 tokens).

Each merge step adds exactly one new token to the vocabulary. If you start with 256 base tokens (byte-level) and perform 31,744 merges, you get a vocabulary of 32,000.

### Encoding New Text

Once trained, encoding a new string applies the learned merge rules in priority order (earliest learned merges first):

1. Split the input into individual characters (or bytes).
2. Scan for the highest-priority merge pair.
3. Apply it everywhere in the sequence simultaneously.
4. Repeat until no more merges apply.

For example, encoding "lowest" with our trained merges:
- Start: `['l', 'o', 'w', 'e', 's', 't']`
- Apply `e + s -> es`: `['l', 'o', 'w', 'es', 't']`
- Apply `es + t -> est`: `['l', 'o', 'w', 'est']`
- No more applicable merges. Final tokens: `['l', 'o', 'w', 'est']`

### The Role of Vocabulary Size

Vocabulary size $|V|$ is a critical hyperparameter that controls the granularity of tokenization:

- **Small vocabulary** (e.g., 8K): More tokens per text, shorter sequences less likely. Lower memory for embedding matrix: $|V| \times d$ where $d$ is embedding dimension. Better generalization to rare words.
- **Large vocabulary** (e.g., 128K+): Fewer tokens per text (better compression), each token carries more information. Larger embedding matrix. Common words and phrases get dedicated tokens, but many tokens receive sparse training signal.

The compression ratio $C$ can be expressed as:

$$C = \frac{\text{number of characters in text}}{\text{number of tokens produced}}$$

Typical compression ratios for English are 3.5-4.5 characters per token with modern BPE tokenizers.

## Why It Matters

BPE is not just one algorithm among equals -- it is the foundation of tokenization for the vast majority of state-of-the-art language models. Understanding BPE is essential because:

- It explains **why models see text the way they do**. When GPT-4 fails at counting letters, the root cause is often BPE tokenization hiding character boundaries.
- It determines **multilingual efficiency**. BPE trained predominantly on English creates tokens that are highly efficient for English but fragment other languages into many small pieces.
- It directly affects **model economics**. Every merge rule that compresses common English phrases into single tokens means fewer tokens processed, less compute consumed, and lower API costs.
- It shapes the **vocabulary** that defines the input and output space of the model.

## Key Technical Details

- **Byte-level BPE**: GPT-2 introduced byte-level BPE, where the base vocabulary is the 256 possible byte values rather than Unicode characters. This guarantees that any input can be encoded (no `<UNK>` tokens) and avoids the massive base vocabulary that Unicode would require (~150,000 characters).
- **Pre-tokenization**: Before BPE is applied, text is typically split using regex rules. GPT-2 uses a pattern that splits on whitespace, punctuation, and contractions. GPT-4's cl100k tokenizer uses a more refined pattern that also splits digits individually. This prevents merges across word boundaries (e.g., "dog" at the end of one word merging with "c" at the start of the next).
- **tiktoken**: OpenAI's open-source tokenizer library implements BPE with optimized Rust code. It's dramatically faster than the original Python implementations. The encoding name `cl100k_base` is used by GPT-4 and `o200k_base` by GPT-4o, indicating approximate vocabulary sizes of 100K and 200K.
- **Merge priority**: During encoding, merges are applied in the order they were learned during training. The first merge learned is the highest-frequency pair and gets applied first. This greedy deterministic process ensures consistent tokenization.
- **Training corpus matters**: If BPE is trained on English Wikipedia, it learns English-centric merges. Training on a multilingual corpus distributes the vocabulary budget across languages, improving multilingual compression at the cost of English compression.

## Common Misconceptions

- **"BPE finds the optimal segmentation."** BPE is a greedy algorithm. It finds a good segmentation, not the optimal one. The Unigram model approach uses a probabilistic framework that can consider global optimality.
- **"BPE merges are applied sequentially to each occurrence."** During encoding, the highest-priority applicable merge is found and applied to all occurrences simultaneously before moving to the next merge.
- **"Byte-level means the model processes raw bytes."** Byte-level BPE still produces subword tokens; it simply uses bytes as the starting alphabet instead of characters. The final tokens are usually multi-byte sequences, not individual bytes.
- **"The same word always produces the same tokens regardless of context."** With standard BPE this is true for the same word in isolation, but pre-tokenization splits can cause different tokenizations depending on surrounding context (e.g., whether a space precedes the word).

## Connections to Other Concepts

- **Tokenization**: BPE is a specific algorithm within the broader category of tokenization methods.
- **Vocabulary Design**: BPE training is the mechanism by which vocabulary is constructed; vocabulary size is the stopping criterion for BPE merges.
- **Token Embeddings**: Each BPE token ID maps to a row in the embedding matrix. The vocabulary BPE produces defines the dimensions of this matrix.
- **Special Tokens**: BPE vocabularies are augmented with special tokens (BOS, EOS, etc.) that are added separately, not learned through the merge process.
- **Context Window**: BPE's compression efficiency determines how many words of text fit within a model's fixed token-length context window.

## Further Reading

- Gage, P. (1994). "A New Algorithm for Data Compression." *The C Users Journal.* -- The original BPE algorithm for data compression.
- Sennrich, R., Haddow, B., & Birch, A. (2016). "Neural Machine Translation of Rare Words with Subword Units." *ACL 2016.* -- The seminal paper adapting BPE to NLP.
- OpenAI. "tiktoken." GitHub repository. -- The production BPE implementation used by GPT models, with detailed documentation of encoding schemes and pre-tokenization patterns.
