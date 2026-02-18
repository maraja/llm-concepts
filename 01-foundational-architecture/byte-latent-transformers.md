# Byte Latent Transformers

**One-Line Summary**: Byte Latent Transformers (BLT) are a tokenizer-free architecture that operates directly on raw bytes with dynamic patching, eliminating tokenization artifacts while matching the performance of token-based models at equivalent compute budgets.

**Prerequisites**: Transformer architecture, tokenization (BPE/WordPiece), attention mechanisms, encoder-decoder architecture, entropy and information theory basics

## What Is Byte Latent Transformers?

Imagine reading a book where every page has been pre-cut into fixed jigsaw puzzle pieces before you see it. Some pieces split words in half, others merge unrelated fragments, and words in different languages get sliced into wildly different numbers of pieces. This is what tokenization does to text. Now imagine instead reading the raw letters directly, but having an intelligent assistant who groups letters into natural, variable-sized chunks based on how surprising or complex each region is -- spending more time on dense technical passages and breezing through predictable boilerplate. That is the Byte Latent Transformer.

BLT, introduced by Meta FAIR (Pagnoni et al., 2024), is a radical departure from the tokenizer-dependent paradigm that has dominated language modeling since the introduction of BPE. Instead of converting text to a fixed vocabulary of subword tokens, BLT operates directly on raw UTF-8 bytes (256 possible values plus special tokens). But it does not naively process every byte through a massive transformer -- that would be computationally prohibitive. Instead, it uses a three-component architecture with dynamic patching to achieve efficiency comparable to token-based models.

The result is a model free from all tokenization artifacts: no more whitespace sensitivity, no more inconsistent number handling, no more poor performance on rare words or non-English scripts, and no more adversarial attacks that exploit tokenizer boundaries. BLT matches the performance of tokenizer-based models at equivalent compute budgets while gaining robustness that fixed vocabularies fundamentally cannot provide.

## How It Works

### The Three-Component Architecture

BLT consists of three distinct modules that work together:

```
Raw bytes: [72, 101, 108, 108, 111, 32, 119, 111, 114, 108, 100]
            H    e    l    l    o   ' '   w    o    r    l    d

         +------------------+
         | 1. Local Encoder |  (lightweight, processes every byte)
         +------------------+
                  |
         Groups bytes into variable-length patches
                  |
         +---------------------+
         | 2. Global Transformer|  (large, processes patch representations)
         +---------------------+
                  |
         Patch-level hidden states
                  |
         +------------------+
         | 3. Local Decoder  |  (lightweight, predicts individual bytes)
         +------------------+
                  |
         Predicted bytes
```

**Local Encoder**: A small, efficient model (e.g., a few transformer layers or a convolutional network) that processes every byte and groups them into variable-length patches. Each patch is represented as a single vector, compressing the byte sequence into a shorter sequence of patch representations. This is where the critical dynamic patching happens.

**Global Transformer**: The main computational workhorse -- a large transformer (comparable in size to a standard LLM) that operates on the patch representations. Because patches are much fewer than raw bytes, this maintains computational efficiency similar to operating on tokens. The global transformer captures long-range dependencies and complex reasoning at the patch level.

**Local Decoder**: Another lightweight model that takes the global transformer's patch-level hidden states and decompresses them back into byte-level predictions. It generates the output byte by byte within each patch, conditioned on the patch representation from the global transformer.

### Entropy-Based Dynamic Patching

The most innovative aspect of BLT is how it decides where to place patch boundaries. Rather than using fixed-size groupings, BLT uses the entropy (uncertainty) of the byte stream to determine boundaries:

```
Input:  "The cat sat on the xylophone"

Entropy: low low low low low low HIGH HIGH HIGH HIGH HIGH HIGH
         T   h   e  ' '  c   a    x    y    l    o    p    h

Patches: [The cat sat on the] [xy] [lo] [phone]
         ^^^^^^^^^^^^^^^^^^^^  ^^^  ^^^  ^^^^^^
         one large patch       smaller patches for rare word
         (predictable text)    (surprising/complex text)
```

When the next byte is highly predictable (low entropy), many bytes are grouped into a single patch. When the next byte is surprising or uncertain (high entropy), patches are smaller, giving the global transformer more positions to attend to and more compute to allocate. This is a form of adaptive computation: the model automatically spends more thinking on hard parts and less on easy parts.

This contrasts sharply with fixed tokenizers, where "the" always gets one token and "xylophone" might get three tokens regardless of context. BLT's patching is context-dependent -- the same word might be grouped differently depending on how predictable it is in a given context.

### Handling Multiple Languages and Scripts

Because BLT operates on UTF-8 bytes, it handles any script or language natively:

```
English:  "Hello"     -> 5 bytes   -> ~1-2 patches
Japanese: "こんにちは"  -> 15 bytes  -> ~3-5 patches
Arabic:   "مرحبا"      -> 10 bytes  -> ~2-3 patches
Emoji:    "🎉"         -> 4 bytes   -> ~1 patch
```

Token-based models give disproportionate representation to English (where common words are single tokens) while fragmenting other languages into many subword pieces, creating a fundamental multilingual bias. BLT's byte-level approach treats all scripts equally, with compute allocation driven by actual complexity rather than vocabulary statistics.

## Why It Matters

1. **Eliminates tokenization artifacts**: No more inconsistent number tokenization ("380" vs "3" "80"), whitespace sensitivity, or poor rare-word handling that plagues every tokenizer-based model.
2. **Adaptive computation**: Dynamic patching allocates more model capacity to complex text regions and less to predictable ones -- something fixed tokenizers fundamentally cannot achieve.
3. **Robustness to perturbations**: BLT is significantly more robust to typos, character-level adversarial attacks, and novel word formations because it processes raw characters rather than relying on a fixed vocabulary.
4. **True multilingual equality**: All languages and scripts are processed through the same mechanism without vocabulary-driven bias toward high-resource languages.
5. **Eliminates tokenizer maintenance**: No need to train, version, or update tokenizers. The model handles any byte sequence, including future Unicode additions, mixed-language text, and binary data.

## Key Technical Details

- BLT matches tokenizer-based model performance at equivalent FLOPs budgets, as demonstrated through extensive scaling experiments up to 8B parameter equivalent models.
- The local encoder and decoder are approximately 10-20x smaller than the global transformer, keeping overhead manageable.
- Dynamic patch sizes typically range from 2-8 bytes, averaging around 4-5 bytes per patch (similar to the ~4 characters/token ratio of BPE tokenizers).
- The entropy threshold for patch boundaries is a tunable hyperparameter that controls the trade-off between sequence length (efficiency) and granularity (expressiveness).
- Training requires byte-level cross-entropy loss at the decoder, but the dominant compute cost remains the global transformer operating on patches.
- BLT demonstrates particular advantages on tasks involving character-level understanding (spelling, character counting, phonetics) where tokenized models struggle.
- The architecture is compatible with standard training infrastructure (mixed-precision training, gradient checkpointing, distributed training) with minor modifications.

## Common Misconceptions

- **"Byte-level models are too expensive to train because sequences are 3-4x longer."** BLT's dynamic patching compresses byte sequences to roughly the same length as token sequences. The global transformer (which dominates compute) operates on patches, not raw bytes. Only the lightweight local encoder and decoder process individual bytes.

- **"Removing the tokenizer means losing all subword information."** The local encoder learns to build patch representations that capture subword structure organically. It effectively learns a context-dependent, soft tokenization that can be more informative than a fixed vocabulary.

- **"BLT is only useful for non-English languages."** While multilingual equity is a significant benefit, BLT also improves robustness, character-level reasoning, and handling of numbers, code, and structured data in English. The advantages are universal.

- **"This is just character-level modeling rebranded."** Previous character-level models processed every character through the full model, making them prohibitively expensive. BLT's three-component architecture with dynamic patching is architecturally novel and achieves fundamentally different efficiency characteristics.

## Connections to Other Concepts

- **Tokenization (BPE, WordPiece, SentencePiece)**: BLT is a direct alternative to all subword tokenization methods, aiming to replace rather than complement them.
- **Adaptive Computation**: Dynamic patching is a form of adaptive computation where the model allocates variable processing to different input regions based on complexity.
- **Encoder-Decoder Architecture**: BLT's three-component structure echoes encoder-decoder designs, with the local encoder/decoder serving as compression/decompression layers around the global transformer.
- **Multi-Token Prediction**: MTP and BLT both aim to improve representations -- MTP through richer training objectives, BLT through more flexible input processing. They could potentially be combined.
- **Mixture of Experts**: Both MoE and BLT's dynamic patching are forms of conditional computation -- allocating resources where they are most needed rather than uniformly.

## Further Reading

- Pagnoni et al., "Byte Latent Transformer: Patches Scale Better Than Tokens" (2024) -- the foundational BLT paper from Meta FAIR
- Xue et al., "ByT5: Towards a Token-Free Future with Pre-trained Byte-to-Byte Models" (2022) -- earlier byte-level approach using a fixed architecture without dynamic patching
- Clark et al., "Canine: Pre-training an Efficient Tokenization-Free Encoder" (2022) -- another tokenizer-free approach using character-level hashing
- Yu et al., "MegaByte: Predicting Million-Byte Sequences with Multiscale Transformers" (2023) -- related multi-scale architecture for long byte sequences
