# Watermarking for LLM-Generated Text

**One-Line Summary**: LLM text watermarking embeds statistically detectable but human-imperceptible signals into generated text by biasing the token selection process during generation, enabling reliable identification of AI-generated content without altering the perceived quality of the text.

**Prerequisites**: Understanding of how autoregressive language models generate text token by token, the concept of a logit distribution over vocabulary at each generation step, sampling strategies (temperature, top-p), hash functions, and basic hypothesis testing (p-values).

## What Is LLM Text Watermarking?

As LLMs generate increasingly human-like text, distinguishing AI-generated content from human-written content becomes critical for combating misinformation, academic dishonesty, and fraud. Post-hoc detectors (trained classifiers that analyze text features) have proven unreliable -- they suffer from high false-positive rates, are easily defeated by paraphrasing, and cannot provide statistical guarantees.

*Recommended visual: Watermarking process showing green/red list partitioning of vocabulary and bias injection during sampling — see [Kirchenbauer et al. Watermarking Paper (arXiv:2301.10226)](https://arxiv.org/abs/2301.10226)*


Watermarking takes a fundamentally different approach: instead of detecting AI text after the fact, it embeds a signal into the text during generation. The signal is imperceptible to human readers but statistically detectable by anyone who knows the watermarking scheme. Think of it as invisible ink -- the text reads normally, but under the right "light" (the detection algorithm), the watermark is revealed.

The landmark approach by Kirchenbauer et al. (2023) introduced a practical watermarking scheme based on biasing token selection during generation. The core idea: at each generation step, partition the vocabulary into a "green list" and a "red list" based on a hash of the preceding token(s), then add a bias to the logits of green-list tokens. This makes the model slightly more likely to choose green-list tokens without dramatically changing the output quality. Over a sequence of many tokens, the statistical excess of green-list tokens becomes detectable.

## How It Works


*Recommended visual: Statistical detection of watermarked text using z-score test on green token frequency — see [Kirchenbauer et al. Paper](https://arxiv.org/abs/2301.10226)*

### The Kirchenbauer et al. Scheme

**Setup**: Choose two parameters:
- gamma: the fraction of vocabulary tokens in the green list (typically 0.25 to 0.5)
- delta: the logit bias added to green-list tokens (typically 1.0 to 2.0)

**Generation (Watermarked)**:

For each token position t in the sequence:

1. **Hash the context**: Compute a hash of the preceding token (or preceding k tokens) to create a pseudorandom seed: `seed = hash(token_{t-1})`.

2. **Partition the vocabulary**: Using the seed, pseudorandomly partition the vocabulary V into a green list G (containing gamma * |V| tokens) and a red list R (containing (1 - gamma) * |V| tokens). The partition is deterministic given the seed, so the same context always produces the same partition.

3. **Bias the logits**: Before applying softmax, add delta to the logits of all green-list tokens:
   ```
   logit'_i = logit_i + delta  if token_i in G
   logit'_i = logit_i           if token_i in R
   ```

4. **Sample normally**: Apply the usual sampling strategy (temperature, top-p, top-k) to the biased logits and select the next token.

The result: at each position, the model has a higher probability of selecting green-list tokens. For high-entropy positions (where many tokens are plausible), the bias has a strong effect. For low-entropy positions (where only one or a few tokens are reasonable), the bias has minimal effect because the correct token dominates regardless of the green/red partition.

**Detection**:

Given a text to check and knowledge of the hash function and gamma:

1. For each token position t, recompute the green list G_t using the hash of the preceding token(s).
2. Count how many tokens in the text fall in their respective green lists: `count_green = |{t : token_t in G_t}|`.
3. Under the null hypothesis (human-written text, no watermark), each token has probability gamma of being in its green list by chance. So `count_green ~ Binomial(n, gamma)` where n is the text length.
4. Compute a z-score: `z = (count_green - gamma * n) / sqrt(n * gamma * (1 - gamma))`.
5. If z exceeds a threshold (e.g., z > 4, corresponding to p < 0.00003), the text is classified as watermarked.

The beauty of this approach is its statistical rigor: you can compute exact p-values for the null hypothesis that the text is not watermarked. This provides quantified confidence, unlike heuristic classifiers.

### Formal Properties

**Distortion-free variant**: Aaronson and Kirchner (concurrent work, later published as part of OpenAI's research) proposed a distortion-free watermark using Gumbel noise. Instead of adding a flat bias, they use the watermarking key to generate Gumbel noise for each token, then select the token that maximizes `log(p_i) + g_i` where `p_i` is the original probability and `g_i` is the Gumbel noise. This provably does not change the marginal distribution of any single token, only introducing correlations between tokens that are detectable across a sequence.

**Information-theoretic analysis**: The strength of the watermark depends on the text entropy. Low-entropy text (highly predictable, like code or formulaic language) has fewer viable token choices, leaving less room for the watermark to influence selection. High-entropy text (creative writing, open-ended responses) has many viable tokens at each position, giving the watermark strong influence. The effective watermark strength is proportional to the average conditional entropy per token.

**Unigram watermarks**: A simpler variant uses a fixed green list (not dependent on context). This is easier to implement and more robust to text reordering, but it is also easier to detect and remove because the green list does not change.

### Token-Level vs. Sequence-Level

The Kirchenbauer scheme operates at the token level -- each token's green/red assignment depends on the preceding context. This creates a chain dependency that makes the watermark robust to local edits but detectable even in partial text. Sequence-level watermarks (embedding a signal across the entire generation) offer different robustness/detectability trade-offs and are an active research area.

## Why It Matters

Text watermarking is a technical foundation for AI governance and content provenance:

- **Academic integrity**: Universities and educational institutions need reliable methods to identify AI-generated submissions. Watermarking provides statistical confidence that heuristic detectors cannot match.
- **Content provenance**: As AI-generated content floods the internet, watermarking enables tracking content back to its source model and deployment, supporting accountability.
- **Regulatory compliance**: Emerging regulations (EU AI Act, proposed US legislation) may require AI-generated content to be identifiable. Watermarking is the most technically viable approach for text.
- **Misinformation defense**: Watermarking allows platforms to flag AI-generated content, helping users assess credibility.

Google DeepMind's SynthID-Text (2024) deployed watermarking in Gemini, representing the first large-scale production deployment of LLM text watermarking. This demonstrated that watermarking can be applied at scale without degrading user-perceived text quality.

## Key Technical Details

- **Minimum text length**: Watermark detection requires sufficient text for statistical significance. With gamma = 0.5 and delta = 2.0, approximately 25-50 tokens are needed for reliable detection (z > 4). Shorter texts produce inconclusive results.
- **Quality impact**: With well-tuned parameters (delta = 1.0-2.0, gamma = 0.25-0.5), the quality impact is minimal. Human evaluators in the Kirchenbauer et al. study could not reliably distinguish watermarked from non-watermarked text. Perplexity increases by 1-5% -- statistically measurable but perceptually insignificant.
- **Context window for hashing**: Using only the immediately preceding token (k=1) for the hash is simplest but makes the watermark more vulnerable to substitution attacks. Using longer contexts (k=2 to 4) improves robustness but can create issues with text that has different preceding context at detection time (e.g., if the text is excerpted from a larger passage).
- **Key management**: The watermarking scheme's security depends on the hash function / key. If the key is public, anyone can detect the watermark (useful for transparency) but also anyone can attempt to remove it. If the key is secret, only the deployer can detect the watermark, but this creates a trust and verification problem.
- **Green-list fraction (gamma)**: Lower gamma (e.g., 0.25) creates a stronger statistical signal per token (because the expected fraction under null hypothesis is lower, so excess green tokens are more surprising) but restricts the model's output more. Higher gamma (e.g., 0.5) is less restrictive but requires more tokens for detection.
- **Multi-bit watermarks**: Advanced schemes encode multiple bits of information (not just presence/absence of a watermark), enabling identification of the specific model, deployment, or even user session that generated the text.

## Robustness and Attacks

**Paraphrasing attacks**: An attacker can pass watermarked text through a non-watermarked LLM to paraphrase it, removing the watermark. This is the primary attack vector and remains partially effective, though strong watermarks (high delta) survive moderate paraphrasing.

**Token substitution**: Replacing a fraction of tokens can reduce the z-score below the detection threshold. However, replacing enough tokens to defeat the watermark typically degrades text quality noticeably.

**Cropping**: Using only a portion of the watermarked text reduces the number of tokens available for detection, potentially making the watermark undetectable. Longer original texts are more robust to cropping.

**Emoji/homoglyph attacks**: Inserting invisible characters or replacing characters with visually similar Unicode characters can break the token-level detection. Pre-processing (normalization) at detection time mitigates this.

**Adaptive attacks**: If the attacker knows the watermarking scheme (but not the key), they can attempt to estimate the green list and substitute green tokens with red tokens. This is harder when the key is secret and the context window is longer.

The fundamental trade-off: **stronger watermarks are more detectable but more distortive and more vulnerable to quality-preserving removal; weaker watermarks are less distortive but require more text for detection.**

## Common Misconceptions

**"Watermarking is the same as AI text detection."** Watermarking is an active measure applied during generation. AI text detection is a passive analysis of existing text. Watermarking provides statistical guarantees; detection classifiers do not. They are fundamentally different approaches.

**"Watermarking makes text obviously AI-generated to human readers."** With appropriate parameters, watermarked text is indistinguishable from non-watermarked text to human readers. The signal is statistical, not perceptual -- it requires counting token-level patterns that humans cannot perceive.

**"Watermarks are unbreakable."** All current watermarking schemes can be removed by sufficiently motivated attackers (e.g., through paraphrasing with a non-watermarked model). The goal is to make removal costly enough that it deters casual misuse, not to create an unbreakable cryptographic seal.

**"Watermarking requires modifying the model."** The Kirchenbauer scheme modifies only the sampling process (logit biasing), not the model weights. It can be applied as a post-processing layer on any LLM without retraining.

**"Short texts can be reliably watermarked."** Below approximately 25-50 tokens, the statistical signal is too weak for reliable detection. Single sentences or short paragraphs often cannot be watermarked reliably.

## Connections to Other Concepts

- **LLM text generation / sampling**: Watermarking modifies the sampling process, making it directly connected to temperature, top-p, top-k, and other sampling strategies.
- **AI safety and alignment**: Watermarking is a safety tool that enables accountability and content provenance for AI-generated text.
- **Guardrails**: Watermarking complements output guardrails -- guardrails filter harmful content, watermarking enables traceability of all content.
- **Hallucination**: Watermarking does not address hallucination directly, but knowing that text is AI-generated (via watermark detection) may prompt users to verify claims more carefully.
- **Red teaming**: Robustness testing of watermarking schemes is a form of red teaming -- adversaries try to remove watermarks while preserving text quality.

## Further Reading

- Kirchenbauer, J. et al. (2023). "A Watermark for Large Language Models." *ICML 2023.* (arXiv: 2301.10226) The foundational paper introducing green-list/red-list watermarking with rigorous statistical analysis.
- Kirchenbauer, J. et al. (2023). "On the Reliability of Watermarks for Large Language Models." (arXiv: 2306.04634) Follow-up analyzing robustness to various attacks and practical deployment considerations.
- Aaronson, S. (2022). Blog post: "My AI Safety Lecture at UT Austin." Describes the Gumbel-noise distortion-free watermarking scheme developed with OpenAI.
- Dathathri, S. et al. (2024). "Scalable Watermarking for Identifying Large Language Model Outputs." *Nature.* The SynthID-Text paper from Google DeepMind, demonstrating production-scale watermarking in Gemini.
- Christ, M. et al. (2024). "Undetectable Watermarks for Language Models." *COLT 2024.* Theoretical work on provably undetectable watermarks.
