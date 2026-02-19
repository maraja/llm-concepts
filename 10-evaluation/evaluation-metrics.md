# Traditional NLP Metrics: BLEU, ROUGE & BERTScore

**One-Line Summary**: BLEU, ROUGE, and BERTScore are automated text evaluation metrics that compare generated text against reference text using n-gram overlap (BLEU, ROUGE) or contextual embedding similarity (BERTScore), each with distinct strengths and well-known limitations.

**Prerequisites**: Understanding of n-grams (contiguous sequences of n words), precision and recall from information retrieval, basic familiarity with word embeddings and contextual representations (e.g., BERT), awareness of NLP tasks like machine translation and text summarization.

## What Are These Metrics?

Imagine you ask three different people to summarize a news article. Each summary will be different -- different word choices, different sentence structures, maybe different aspects emphasized -- but they could all be equally good. Now imagine you need an automated system to score a fourth summary. How do you decide if it is good?

*Recommended visual: BLEU n-gram matching example showing precision calculation between generated and reference text — see [Hugging Face Evaluate Documentation](https://huggingface.co/docs/evaluate/index)*


The fundamental challenge of text evaluation is that there are many valid ways to express the same meaning. Traditional metrics address this by comparing generated text against one or more "reference" texts (human-written gold standards) and measuring how much they overlap.

BLEU asks: "How many chunks of the generated text also appear in the reference?" (precision-focused). ROUGE asks: "How many chunks of the reference also appear in the generated text?" (recall-focused). BERTScore asks: "How semantically similar are the generated and reference texts at the token level?" (meaning-focused).

Think of it like comparing two recipes for chocolate cake. BLEU checks whether the ingredients in your recipe are real cake ingredients. ROUGE checks whether your recipe covers all the important cake ingredients. BERTScore checks whether your recipe would actually produce something that tastes like chocolate cake, even if you used "cocoa powder" where the reference said "dark chocolate."

## How It Works


*Recommended visual: BERTScore computation showing cosine similarity between contextual embeddings of generated and reference tokens — see [BERTScore Paper (arXiv:1904.09675)](https://arxiv.org/abs/1904.09675)*

### BLEU (Bilingual Evaluation Understudy)

BLEU was designed in 2002 for machine translation evaluation. It computes the precision of n-gram matches between a candidate translation and one or more reference translations.

**Step-by-step calculation**:

1. **Modified n-gram precision**: For each n-gram size (typically 1 through 4), count how many n-grams in the candidate appear in any reference. The "modified" part is crucial: each reference n-gram can only be "used" as many times as it appears in the reference, preventing a candidate that repeats one correct word from getting perfect precision.

```
p_n = (number of clipped matching n-grams) / (total n-grams in candidate)
```

2. **Brevity Penalty (BP)**: BLEU is precision-based, so a very short candidate could achieve high precision by only including words it is confident about. The brevity penalty discourages this:

```
BP = exp(1 - r/c)    if c < r
BP = 1                if c >= r

where c = candidate length, r = reference length
```

3. **Final BLEU score**: Combine the n-gram precisions (typically using geometric mean) with the brevity penalty:

```
BLEU = BP * exp( sum_{n=1}^{N} w_n * log(p_n) )

where w_n = 1/N (uniform weights, typically N=4)
```

BLEU scores range from 0 to 1 (often reported as 0-100). A score of 1.0 means the candidate perfectly matches a reference. In practice, human translations of professional quality typically score 0.30-0.50 on BLEU, and scores above 0.60 are rare.

**BLEU variants**: SacreBLEU standardized the tokenization and implementation details that caused BLEU scores to be non-comparable across papers. It is now the recommended implementation for research.

### ROUGE (Recall-Oriented Understudy for Gisting Evaluation)

ROUGE was designed in 2004 for summarization evaluation. While BLEU measures precision (how much of the candidate is correct), ROUGE emphasizes recall (how much of the reference is covered).

**ROUGE-N**: Measures n-gram recall between candidate and reference:

```
ROUGE-N = (number of matching n-grams) / (total n-grams in reference)
```

The most commonly reported variants are ROUGE-1 (unigram recall) and ROUGE-2 (bigram recall).

**ROUGE-L**: Uses the Longest Common Subsequence (LCS) between candidate and reference. Unlike n-gram methods, LCS does not require consecutive matches, allowing it to capture sentence-level structure more flexibly:

```
R_lcs = LCS(candidate, reference) / len(reference)    (recall)
P_lcs = LCS(candidate, reference) / len(candidate)     (precision)
ROUGE-L = F1(R_lcs, P_lcs)
```

**ROUGE-Lsum**: A variant for multi-sentence summaries that computes ROUGE-L at the sentence level and aggregates, avoiding cross-sentence LCS matches that would be meaningless.

ROUGE scores also range from 0 to 1. Typical ROUGE-1 scores for competitive summarization systems are 0.40-0.50 on standard benchmarks like CNN/DailyMail.

### BERTScore

BERTScore, introduced in 2020, addresses the fundamental limitation of BLEU and ROUGE: they only detect exact token matches. "Doctor" and "physician" are treated as completely different words, even though they are near-synonyms.

**Step-by-step calculation**:

1. **Encode both texts**: Pass the candidate and reference through a pre-trained contextual model (typically RoBERTa-large) to obtain token-level embeddings. Crucially, these are contextual embeddings -- the representation of each word depends on its surrounding context.

2. **Pairwise cosine similarity**: Compute the cosine similarity between every candidate token embedding and every reference token embedding, producing a similarity matrix.

3. **Greedy matching**:
   - **Recall**: For each reference token, find the most similar candidate token. Average these maximum similarities.
   - **Precision**: For each candidate token, find the most similar reference token. Average these maximum similarities.

```
R_BERT = (1/|ref|) * sum_{r in ref} max_{c in cand} cos_sim(r, c)
P_BERT = (1/|cand|) * sum_{c in cand} max_{r in ref} cos_sim(r, c)
F_BERT = 2 * (P_BERT * R_BERT) / (P_BERT + R_BERT)
```

4. **Importance weighting (optional)**: Tokens can be weighted by inverse document frequency (IDF) to emphasize content words over function words.

BERTScore values are typically in the range of 0.85-0.95 for reasonable text, making raw scores hard to interpret. The authors recommend using baseline rescaling to normalize scores to a 0-1 range relative to a corpus baseline.

## Why It Matters

These metrics serve different roles in the modern NLP ecosystem:

**BLEU** remains the standard metric for machine translation research, despite its limitations. Decades of published BLEU scores provide a historical baseline for measuring progress. When a paper reports improvements in machine translation, BLEU scores are still expected.

**ROUGE** remains standard for summarization evaluation. While researchers acknowledge its limitations, it provides a fast, cheap, reproducible signal for comparing summarization systems. Most summarization papers report ROUGE-1, ROUGE-2, and ROUGE-L.

**BERTScore** has become the preferred automated metric when semantic fidelity matters more than exact wording. It is particularly useful for evaluating paraphrase generation, text simplification, and other tasks where surface-level diversity is expected.

However, in the era of large language models, all three metrics have diminished importance for several reasons:

- LLM outputs are often open-ended, with no single "correct" reference to compare against.
- The quality dimensions that matter most (helpfulness, safety, instruction-following) are not captured by text overlap.
- LLM-as-a-Judge has largely replaced these metrics for evaluating chat-oriented models.

These metrics remain most relevant for narrower, well-defined tasks (translation, summarization) and for researchers who need cheap, reproducible evaluation during development iterations.

## Key Technical Details

- BLEU is a corpus-level metric by design. Computing BLEU on a single sentence is statistically unreliable and can produce degenerate results. Sentence-level smoothed BLEU variants exist but are approximations.
- ROUGE, unlike BLEU, is typically computed per-example and then averaged across the corpus. This subtle difference means the two metrics are not directly comparable.
- BERTScore's quality depends heavily on the underlying model. RoBERTa-large is the default, but domain-specific models (SciBERT for scientific text, CodeBERT for code) may be more appropriate in specialized domains.
- All three metrics require reference texts, which limits their applicability. Reference-free evaluation is an active research area.
- BLEU and ROUGE are tokenization-sensitive. Different tokenization choices (word-level, subword, with or without punctuation normalization) can change scores by several points.
- None of these metrics penalize factual errors if the phrasing overlaps with the reference. A summary that gets a key fact wrong but uses similar words to the reference will still score well.

## Common Misconceptions

- **"High BLEU/ROUGE means high quality."** These metrics measure surface overlap, not quality. A generated text can score high by copying phrases from the reference while being incoherent or factually wrong. Conversely, an excellent paraphrase using entirely different words will score near zero.
- **"BLEU measures how good a translation is."** BLEU measures how similar a translation is to specific reference translations. It penalizes valid alternative translations that happen to use different words or structures.
- **"BERTScore solves the problems of BLEU and ROUGE."** BERTScore improves on surface-level matching but still only captures semantic similarity to a reference. It does not evaluate factual accuracy, coherence, or task-specific quality.
- **"These metrics are outdated and useless."** For well-defined tasks with clear references (translation, summarization), they remain useful as fast, cheap development metrics. They are less useful for evaluating general-purpose LLMs.
- **"ROUGE is just BLEU for summarization."** While conceptually related, ROUGE and BLEU differ fundamentally: ROUGE emphasizes recall (coverage of reference content), while BLEU emphasizes precision (correctness of generated content). They answer different questions about text quality.

## Connections to Other Concepts

- **Perplexity**: An intrinsic metric (measures model behavior) whereas BLEU, ROUGE, and BERTScore are extrinsic metrics (measure output quality against references). They complement each other.
- **LLM-as-a-Judge**: Has largely replaced these metrics for evaluating open-ended LLM outputs, but still sometimes uses them as features in a broader evaluation rubric.
- **Tokenization**: All three metrics are sensitive to tokenization choices. BLEU and ROUGE operate on token sequences; BERTScore uses the underlying model's tokenizer.
- **Embeddings and Representations**: BERTScore relies on contextual embeddings, connecting it to the broader literature on learned text representations.
- **Benchmarks**: Many standard benchmarks (WMT for translation, CNN/DailyMail for summarization) report these metrics as their primary evaluation measure.
- **Human Evaluation**: The ultimate validation for all automated metrics. A metric's value is measured by its correlation with human judgments.

## Further Reading

- Papineni et al., "BLEU: a Method for Automatic Evaluation of Machine Translation" (2002) -- the original BLEU paper, one of the most cited papers in NLP.
- Lin, "ROUGE: A Package for Automatic Evaluation of Summaries" (2004) -- the foundational ROUGE paper that standardized summarization evaluation.
- Zhang et al., "BERTScore: Evaluating Text Generation with BERT" (2020) -- introduces BERTScore and demonstrates its superior correlation with human judgments compared to n-gram metrics.
