# Tokenization

**One-Line Summary**: Tokenization is the process of breaking raw text into discrete units (tokens) that a language model can process numerically, and the choices made here ripple through every aspect of model behavior.

**Prerequisites**: Basic understanding of what a language model does (predicts next tokens), familiarity with the idea that neural networks operate on numbers rather than raw text.

## What Is Tokenization?

Imagine you're trying to teach someone a new language, but they can only learn by memorizing flashcards. Each flashcard has a chunk of text on one side and a number on the other. The question is: what should each flashcard contain? A single letter? A whole word? Something in between?

![Comparison of tokenization strategies: character-level, word-level, and subword (BPE/WordPiece) showing the trade-offs between vocabulary size and sequence length](https://huggingface.co/datasets/huggingface-course/documentation-images/resolve/main/en/chapter6/tokenization_strategies.svg)
*Source: [Hugging Face NLP Course – Chapter 6: Tokenizers](https://huggingface.co/learn/nlp-course/chapter6/1)*


Tokenization is the answer to that question for language models. It's the translation layer between human-readable text and the numerical IDs that a model actually processes. When you type "Hello, world!" into ChatGPT, the model never sees those characters directly. Instead, a tokenizer breaks the text into pieces -- tokens -- and converts each piece into an integer. The model thinks entirely in these integers.

This seemingly mundane preprocessing step is one of the most consequential design decisions in all of NLP. It determines how much text fits in a model's context window, how well the model handles different languages, why LLMs struggle with simple arithmetic, and how much each API call costs you.

## How It Works


![Overview of the three main subword tokenization algorithms: BPE (bottom-up merging), WordPiece (likelihood-based merging), and Unigram (top-down pruning)](https://huggingface.co/datasets/huggingface-course/documentation-images/resolve/main/en/chapter6/bpe_subword.svg)
*Source: [Hugging Face NLP Course – Subword Tokenization](https://huggingface.co/learn/nlp-course/chapter6/5)*

### Why Not Just Use Characters?

The simplest approach would be to treat each character as a token. English has roughly 100 common characters (letters, digits, punctuation), so the vocabulary would be tiny. The problem: the sequence "understanding" becomes 13 tokens. A 4096-token context window would hold only about 4,000 characters -- roughly one page of text. The model would also need to learn spelling from scratch, wasting capacity on something trivially solved by other approaches.

![BPE tokenization process showing how text is split into subword tokens of varying granularity](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/tokenizers/bpe_tokenization.png)
*Source: [Hugging Face – Tokenizers Documentation](https://huggingface.co/docs/tokenizers/index)*


### Why Not Just Use Words?

The opposite extreme is word-level tokenization. This is compact but creates an explosion in vocabulary size. English alone has over 170,000 words in active use, plus proper nouns, technical jargon, and misspellings. Any word not in the vocabulary becomes an unknown token (`<UNK>`), making the model unable to process novel words. Morphologically rich languages like Turkish or Finnish, where a single "word" can encode an entire sentence, make this approach completely untenable.

### The Subword Compromise

Modern tokenizers split text into **subword units** -- pieces that are larger than characters but smaller than (or equal to) words. Common words like "the" get their own token, while rare words are decomposed into meaningful pieces.

Consider how "unhappiness" might be tokenized by different methods:

- **Character-level**: `u`, `n`, `h`, `a`, `p`, `p`, `i`, `n`, `e`, `s`, `s` (11 tokens)
- **BPE (GPT-style)**: `un`, `happiness` or `un`, `happ`, `iness` (2-3 tokens)
- **WordPiece (BERT-style)**: `un`, `##happi`, `##ness` (3 tokens, `##` indicates continuation)
- **Word-level**: `unhappiness` (1 token, if in vocabulary) or `<UNK>` (if not)

The subword approach elegantly captures morphological structure: "un-" as a negation prefix, "happi-" as the root, and "-ness" as a nominalizing suffix.

### The Major Algorithms

**Byte-Pair Encoding (BPE)**: Starts with individual characters (or bytes) and iteratively merges the most frequent adjacent pair. Used by GPT-2, GPT-3, GPT-4, LLaMA, and most modern LLMs. It is bottom-up and greedy.

*See also the interactive tokenizer visualization at: [Tiktokenizer](https://tiktokenizer.vercel.app/) -- lets you compare how different models (GPT-4, LLaMA, etc.) tokenize the same input text.*


**WordPiece**: Similar to BPE but selects merges based on maximizing the likelihood of the training data rather than raw frequency. The merge that most increases $\log P(\text{corpus})$ is chosen. Used by BERT and its variants.

**Unigram Language Model**: Works top-down. Starts with a large vocabulary and iteratively removes tokens whose loss contributes least to the overall corpus likelihood. Assigns probabilities to each token and uses the Viterbi algorithm to find the most probable segmentation. Used by T5 and multilingual models.

**SentencePiece**: Not an algorithm itself but a **framework** that implements both BPE and Unigram. Its key innovation is treating the input as a raw byte stream (including whitespace), making it language-agnostic. It doesn't require pre-tokenization (splitting on spaces first), which is crucial for languages like Chinese and Japanese that don't use spaces.

## Why It Matters

Tokenization's impact is pervasive and often underappreciated:

- **Cost**: API pricing is per-token. A poorly tokenized language might use 3x more tokens for the same content, tripling the cost.
- **Context length**: If your tokenizer produces 2 tokens per word on average instead of 1.3, your effective context window shrinks by ~35%.
- **Multilingual equity**: English text typically tokenizes at ~1.3 tokens per word with GPT-4's tokenizer. Hindi might be 3-4x that. Bengali or Burmese can be 5-10x. This means non-English users get less text in the same context window and pay more per word.
- **Arithmetic**: The number "12345" might tokenize as `123`, `45` -- the model never sees the individual digits as aligned place values, making arithmetic unreliable.
- **Spelling**: Models struggle with character-level tasks ("How many r's in strawberry?") because they literally don't see individual characters.

## Key Technical Details

- Vocabulary sizes in practice range from ~32,000 (LLaMA) to ~200,000 (GPT-4's cl100k, DeepSeek).
- Modern tokenizers are **byte-level**, meaning they can encode any byte sequence, eliminating unknown tokens entirely.
- Tokenization is deterministic at inference time -- the same input always produces the same tokens -- but some frameworks like SentencePiece support stochastic **subword regularization** during training for robustness.
- The tokenizer is trained separately from the model on a (potentially different) text corpus before model training begins. Once fixed, it never changes during model training.
- Pre-tokenization rules (splitting on whitespace, punctuation, digits) are applied before the subword algorithm runs, and these rules significantly affect the final vocabulary.

## Common Misconceptions

- **"Tokens are words."** They are not. Tokens can be subwords, individual characters, punctuation, or whitespace. The word "indistinguishable" is typically 3-4 tokens.
- **"All tokenizers work the same way."** Different models use completely different tokenizers. You cannot reuse GPT-4's tokenizer for LLaMA and expect sensible results.
- **"Tokenization is a solved problem."** Active research continues. Byte-level models (like ByT5 and MambaByte) eliminate tokenization entirely. Others explore dynamic tokenization that adapts to the input.
- **"More vocabulary is always better."** Larger vocabularies mean larger embedding matrices (which consume parameters) and sparser training signal per token. There is a genuine trade-off.

## Connections to Other Concepts

- **Token Embeddings**: Tokenization produces integer IDs; embeddings convert those IDs into the dense vectors the model actually processes.
- **Byte-Pair Encoding**: The dominant specific algorithm behind most modern tokenizers.
- **Vocabulary Design**: The higher-level decisions about vocabulary size and composition that shape tokenizer behavior.
- **Context Window**: Tokenization efficiency directly determines how much text fits in the model's context.
- **Special Tokens**: Tokenizers include special control tokens (BOS, EOS, PAD) that aren't derived from text.
- **Positional Encoding**: Positions are assigned per-token, so tokenization granularity affects what "position" means.

## Further Reading

- Sennrich, R., Haddow, B., & Birch, A. (2016). "Neural Machine Translation of Rare Words with Subword Units." *ACL 2016.* -- The paper that introduced BPE to NLP.
- Kudo, T. & Richardson, J. (2018). "SentencePiece: A simple and language independent subword tokenizer and detokenizer for Neural Text Processing." *EMNLP 2018.* -- The framework behind most modern tokenizers.
- Petrov, A., La Malfa, E., Torr, P., & Biber, A. (2024). "Language Model Tokenizers Introduce Unfairness Between Languages." *NeurIPS 2024.* -- A rigorous study of how tokenization creates cross-lingual inequity.
