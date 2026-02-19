# Training Data Curation

**One-Line Summary**: Training data curation -- the process of collecting, filtering, deduplicating, and mixing massive text datasets -- is arguably the most underappreciated factor in LLM quality, with data quality consistently proving more important than data quantity.

**Prerequisites**: Pre-training basics, tokenization, understanding of web scraping at a high level, basic statistics (hashing, similarity), the concept of model bias and toxicity, scaling laws (the relationship between data and model performance).

## What Is Training Data Curation?

Imagine you are preparing ingredients for a world-class restaurant. You could buy everything from the cheapest wholesale market -- huge quantities for low cost -- but your dishes would taste mediocre because the ingredients are inconsistent, sometimes spoiled, and occasionally contaminated. Alternatively, you could carefully source from trusted suppliers, inspect every delivery, remove anything subpar, and thoughtfully combine ingredients in precise proportions. The second approach costs more effort but produces dramatically better results.

*Recommended visual: The data curation pipeline showing the stages: raw web crawl, text extraction, language filtering, quality filtering, deduplication, toxic content removal, and final data mixing — see [Hugging Face -- FineWeb Dataset Documentation](https://huggingface.co/datasets/HuggingFaceFW/fineweb)*


Training data curation is this sourcing and preparation process for LLMs. The raw internet contains trillions of tokens of text, but most of it is low quality: duplicated content, spam, machine-generated filler, toxic material, personally identifiable information, and text in formats that teach the model little of value. Curating training data means transforming this raw material into a carefully constructed dataset that maximizes what the model learns per token processed.

This is not a solved problem. The quality of the training data is widely believed to be one of the primary differentiators between frontier models and their weaker competitors.

## How It Works


*See the deduplication impact analysis in: [Lee et al., "Deduplicating Training Data Makes Language Models Better" (arXiv:2107.06499)](https://arxiv.org/abs/2107.06499), Figure 1, which shows how removing duplicate content improves model performance on downstream benchmarks while reducing memorization and training compute waste.*

### Stage 1: Raw Data Collection

The foundation of most LLM training datasets is web crawl data:

**Common Crawl**: A nonprofit organization that crawls the web and makes the data publicly available. Common Crawl produces roughly 3-5 billion web pages per monthly crawl, amounting to petabytes of raw HTML. This is by far the largest source of text data for LLM training. However, raw Common Crawl is extremely noisy -- it contains boilerplate, navigation menus, ads, cookie notices, error pages, and vast amounts of near-duplicate content.

Beyond web crawl, datasets typically include:

- **Books**: Digitized books provide high-quality, long-form text with diverse topics. Sources include Books3, Project Gutenberg, and licensed collections. Book data teaches narrative structure, sustained argumentation, and domain expertise.
- **Code**: GitHub repositories, Stack Overflow, and coding documentation. Code data improves not only coding ability but also logical reasoning and structured thinking.
- **Academic papers**: arXiv, Semantic Scholar, PubMed. Provides scientific reasoning, mathematical notation, and technical language.
- **Wikipedia**: Extremely high information density. Nearly every major LLM dataset includes the full Wikipedia dump.
- **Conversational data**: Forums (Reddit), Q&A sites, dialogue transcripts. Teaches conversational patterns and diverse perspectives.

### Stage 2: Text Extraction and Cleaning

Raw HTML must be converted to clean text:

1. **HTML parsing**: Extract the main content, removing navigation, headers, footers, ads, and boilerplate. Tools like trafilatura and jusText are commonly used.
2. **Language identification**: Classify the language of each document. Models like fastText's language identifier can process billions of documents efficiently. Filtering for target languages (typically English-heavy, with proportional representation of other languages).
3. **Format normalization**: Standardize encoding (UTF-8), remove excessive whitespace, fix character encoding errors.

### Stage 3: Quality Filtering

This is where the most consequential decisions are made:

**Heuristic filters**: Rule-based removal of obviously low-quality content:
- Documents that are too short (< 50 words) or too long (likely data dumps).
- Documents with excessive special characters, URLs, or formatting artifacts.
- Documents with abnormally high or low perplexity (measured by a small reference language model).
- Documents with too many repeated lines or paragraphs.
- "Dirty word" ratio filters for extreme content.

**Classifier-based filtering**: Training a quality classifier (often a small language model) to distinguish "high-quality" from "low-quality" text. The definition of quality is subjective and consequential:
- Some teams use Wikipedia and published books as positive examples and random web pages as negative examples.
- This approach biases toward formal, encyclopedic writing, which may not be desirable for all model capabilities.
- The FineWeb-Edu dataset used educational content scoring to filter for text with high instructional value.

**Perplexity filtering**: Using a reference language model (e.g., a small GPT-2) to score documents. Very high perplexity suggests gibberish or machine-generated noise. Very low perplexity suggests repetitive or boilerplate text. A middle range often correlates with well-written natural text.

### Stage 4: Deduplication

Duplicate and near-duplicate content is one of the biggest quality problems in web data. The same article might appear on hundreds of websites. Training on duplicates wastes compute and can cause the model to memorize specific passages rather than learning generalizable patterns.

**Exact deduplication**: Hash each document (or paragraph/line) and remove exact matches. Fast but misses near-duplicates.

**MinHash with Locality-Sensitive Hashing (LSH)**: The standard approach for fuzzy deduplication at scale:
1. Convert each document to a set of n-grams (e.g., 5-grams of words).
2. Compute multiple hash functions over the n-gram sets to create a "MinHash signature" -- a compact fingerprint that approximates the Jaccard similarity between document pairs.
3. Use LSH to efficiently find document pairs with high estimated similarity (e.g., Jaccard > 0.8).
4. Remove or merge duplicates within each cluster.

MinHash/LSH can process billions of documents efficiently because it avoids comparing every pair (which would be $O(n^2)$). The computational complexity is approximately $O(n)$ with appropriate LSH parameters.

**Substring deduplication**: Remove repeated substrings across documents (e.g., common disclaimers, legal boilerplate, website templates). The suffix array approach (used by Lee et al., 2022) identifies shared substrings of length 50+ tokens.

### Stage 5: Toxic Content and PII Removal

- **Toxicity classifiers**: Models trained to detect hate speech, violence, explicit content, and other harmful material. Documents exceeding a toxicity threshold are removed or flagged.
- **PII detection and redaction**: Regular expressions and named entity recognition to identify and remove email addresses, phone numbers, social security numbers, and other personally identifiable information.
- **Copyright-sensitive content**: Increasingly, teams filter or limit content with clear copyright concerns, though practices vary widely and the legal landscape is evolving.

### Stage 6: Data Mixing

The final dataset is a carefully proportioned blend of different sources. The mixing ratios significantly affect model behavior:

| Source | Typical Range | Effect of Increase |
|--------|--------------|-------------------|
| Web (filtered) | 50-70% | Broader language coverage, more diverse knowledge |
| Code | 5-20% | Better coding, improved reasoning |
| Books | 5-15% | Better long-form reasoning, narrative understanding |
| Academic | 5-10% | Scientific knowledge, formal reasoning |
| Wikipedia | 3-5% | Factual density, structured knowledge |
| Multilingual | 5-20% | Multilingual capabilities |

These ratios are typically determined through **ablation studies** -- training smaller models with different mixes and evaluating downstream performance. The optimal mix depends on the intended model use case.

### The Rise of Synthetic Data

As available natural text approaches its limits, synthetic data -- text generated by existing LLMs -- has become increasingly important:

- **Textbook-quality explanations**: Using GPT-4-class models to generate educational content (as in Microsoft's "Textbooks Are All You Need" work for Phi models).
- **Instruction-response pairs**: Generating diverse prompts and high-quality responses for fine-tuning.
- **Code with verified correctness**: Generating code along with test cases, filtering for code that passes tests.
- **Data augmentation**: Paraphrasing, translating, or elaborating on existing text to create diverse variants.

The quality ceiling for synthetic data is the capability of the generating model, creating a bootstrap problem: you need a good model to generate good data to train a better model.

### Notable Datasets

- **The Pile** (EleutherAI, 2020): 825 GB, 22 diverse sources, one of the first large-scale curated datasets for LLM training. Pioneered transparent dataset documentation.
- **RedPajama** (Together AI, 2023): Open reproduction of the LLaMA training data mix, totaling 1.2T tokens across 7 source categories.
- **FineWeb** (HuggingFace, 2024): 15T tokens of filtered Common Crawl data with extensive quality filtering. FineWeb-Edu further filters for educational content.
- **Dolma** (AI2, 2024): 3T tokens with detailed data provenance and documentation, designed for reproducible research.
- **DCLM** (DataComp for Language Models, 2024): A benchmark for dataset curation, providing a standardized framework for comparing data filtering strategies.

### The "Data Wall" Problem

There is growing concern that the available supply of high-quality natural text is approaching its limits. Estimates suggest:

*See also the data mixing ablation results at: [Penedo et al., "The FineWeb Datasets" (arXiv:2406.17557)](https://arxiv.org/abs/2406.17557) -- includes figures comparing model quality across different filtering strategies and data mix proportions, demonstrating that data quality dominates data quantity.*


- Total "high-quality" text on the internet: approximately 5-10 trillion tokens.
- Frontier models already train on 1-15 trillion tokens.
- Multiple organizations are competing for the same finite pool.

This has implications: models may need to train on the same data multiple times (with diminishing returns), synthetic data may become essential, or new modalities (video transcripts, audio, images) may need to be incorporated. Some researchers argue the "data wall" will slow progress; others believe better filtering, synthetic generation, and multi-modal data will provide sufficient scaling.

## Why It Matters

Data curation is where the adage "garbage in, garbage out" applies with full force. Multiple studies have shown that:

- A smaller model trained on higher-quality data can outperform a much larger model trained on lower-quality data.
- The Phi series of models (Microsoft) demonstrated that carefully curated "textbook-quality" data can achieve disproportionately strong performance at small scales.
- Deduplication alone can improve model performance by 5-10% on benchmarks while reducing training compute.

As model architectures converge (most frontier models are dense transformers with similar designs), **data curation has become the primary competitive differentiator**. The details of how organizations filter, mix, and prepare their training data are among their most closely guarded secrets.

## Key Technical Details

- **Data quality > data quantity**: This finding has been replicated across multiple studies and scale regimes. Better filtering consistently beats more data.
- **Deduplication is non-negotiable**: Training on highly duplicated data leads to increased memorization, worse generalization, and wasted compute.
- **The optimal data mix is model-size dependent**: Smaller models may benefit from a higher proportion of high-quality, curated sources, while larger models can better leverage noisier web data.
- **Epoch count matters**: Training for more than 1-2 epochs (passes through the data) shows diminishing returns. Beyond 4 epochs, performance can degrade.
- **Data ordering effects**: The sequence in which data is presented during training can affect convergence, though this is less studied than data composition.
- **Legal and ethical considerations**: Copyright law, privacy regulations (GDPR), and consent issues around training data are increasingly important and unresolved.

## Common Misconceptions

- **"Just scrape the whole internet."** Raw web data is mostly noise. Without extensive filtering, the model learns to generate spam, boilerplate, and gibberish alongside useful text.
- **"More data always helps."** Low-quality data can actively harm performance. A model trained on 1T high-quality tokens can outperform one trained on 10T low-quality tokens.
- **"Data curation is just a preprocessing step."** It is an ongoing research and engineering challenge. The most impactful improvements in model quality often come from data improvements, not architecture changes.
- **"Open-source datasets are as good as proprietary ones."** While open datasets like FineWeb and Dolma are excellent, frontier labs invest enormous resources in proprietary data pipelines that may produce superior training data.
- **"Synthetic data is a complete solution to the data wall."** Synthetic data generated by current models is bounded by those models' capabilities and can introduce systematic biases. It is a complement to, not a replacement for, natural data.

## Connections to Other Concepts

- **Pre-Training**: Data curation determines what the model learns during pre-training.
- **Scaling Laws**: The relationship between data quantity and model performance; data quality shifts the scaling curve.
- **Tokenization**: The tokenizer determines how the curated text is converted to training tokens.
- **Emergent Abilities**: The data mix may influence which capabilities emerge and at what scale.
- **Fine-Tuning**: High-quality curated data is equally important for the supervised fine-tuning stage.
- **Bias and Fairness**: Data curation decisions (what to include, what to filter) directly determine the model's biases.

## Further Reading

- Gao, L., et al. (2020). "The Pile: An 800GB Dataset of Diverse Text for Language Modeling" -- One of the first transparently documented large-scale LLM training datasets, establishing practices for data documentation.
- Lee, K., et al. (2022). "Deduplicating Training Data Makes Language Models Better" -- Demonstrated the significant impact of deduplication on model quality and memorization.
- Penedo, G., et al. (2024). "The FineWeb Datasets: Decanting the Web for the Finest Text Data at Scale" -- Detailed documentation of modern data filtering pipelines and the impact of quality filtering on model performance at scale.
