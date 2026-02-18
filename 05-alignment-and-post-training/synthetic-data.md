# Synthetic Data for Training

**One-Line Summary**: Synthetic data generation uses existing LLMs to create training data for other (often smaller) models, offering a scalable path around the "data wall" but introducing risks of model collapse, reduced diversity, and inherited biases.

**Prerequisites**: Understanding of supervised fine-tuning, instruction tuning, the basics of how LLMs generate text, and the general concept of training data quality and its impact on model performance.

## What Is Synthetic Data for Training?

Natural language training data traditionally comes from human-written text: books, websites, forums, and curated annotations. But as models have grown more capable, a powerful new approach has emerged: using LLMs themselves to generate training data.

This might sound circular -- how can a model train another model? The answer lies in asymmetry. A large, capable model (the "teacher") can generate high-quality examples that a smaller, cheaper model (the "student") learns from. The teacher has already absorbed knowledge from human data; synthetic generation is a form of *knowledge distillation* through data.

Think of it like an expert professor writing a textbook. The professor's knowledge came from decades of reading, research, and experience (analogous to pre-training on human data). The textbook they write is "synthetic" -- they authored it -- but it's a legitimate, valuable learning resource for students. The quality depends entirely on the professor's expertise and how thoughtfully they write.

## How It Works

### Self-Instruct: The Foundational Framework

The Self-Instruct framework (Wang et al., 2023) demonstrated the basic pipeline:

1. **Start with a small seed set** of human-written instruction-response pairs (e.g., 175 examples).
2. **Prompt an LLM** to generate new instructions inspired by (but different from) the seed examples.
3. **Filter** generated instructions for quality, diversity, and uniqueness.
4. **Generate responses** for the new instructions using the LLM.
5. **Repeat** to build a large dataset from a small seed.

This approach was famously used to create the **Alpaca** dataset: Stanford researchers used GPT-3.5 (text-davinci-003) to generate 52,000 instruction-following examples, then fine-tuned LLaMA-7B on them. The resulting model showed remarkably strong instruction-following capability for its size, at a data generation cost of under $500.

### Generating Preference Pairs

For DPO and RLHF training, synthetic data generation extends to creating preference pairs:

1. **Generate multiple responses** to the same prompt using different models, different temperatures, or different prompting strategies.
2. **Use a strong model as judge** to rank the responses (e.g., "Which response is more helpful, accurate, and clear?").
3. **Construct preference pairs** $(y_w, y_l)$ from the rankings.

This approach, sometimes called "LLM-as-a-judge," enables creating large-scale preference datasets without human annotation.

### The Phi-Series Approach: Synthetic Textbooks

Microsoft's Phi series (Phi-1, Phi-1.5, Phi-2, Phi-3) took a different approach. Rather than generating instruction-response pairs, they generated **synthetic textbooks** -- carefully structured educational content designed to teach reasoning and knowledge in a curriculum-like progression.

The key principles:
- **"Textbook-quality" data**: Generated content is structured, clear, and pedagogically sound -- not just factual but explanatory.
- **Exercises and solutions**: The synthetic data includes practice problems with worked solutions, forcing the model to learn problem-solving strategies.
- **Curriculum design**: Topics are organized in a progression from simple to complex, ensuring the model builds understanding incrementally.
- **Diversity through specification**: Rather than hoping for diversity, the generation process explicitly requests content across different topics, difficulty levels, and styles.

The Phi-1 model (1.3B parameters) trained largely on synthetic data achieved performance competitive with models 10x its size on coding benchmarks, demonstrating that data quality can partially substitute for model scale.

### Synthetic Data for Specialized Domains

For domain-specific fine-tuning (medical, legal, financial), synthetic data generation follows a pattern:

1. **Seed with domain knowledge**: Provide the teacher model with domain-specific context (textbooks, guidelines, examples).
2. **Generate domain-specific Q&A pairs**: "Given this medical guideline, generate 20 clinical questions and detailed answers."
3. **Expert validation on a subset**: Have domain experts review a random sample to catch systematic errors.
4. **Fine-tune the student model** on the validated synthetic dataset.

### Evol-Instruct: Evolving Complexity

The WizardLM approach (Xu et al., 2023) introduced **Evol-Instruct**, which systematically increases instruction complexity:

1. Start with a simple instruction.
2. Prompt an LLM to make it more complex through specific transformations:
   - Add constraints ("...but do it without using any loops")
   - Increase reasoning depth ("...and explain why each step is necessary")
   - Add multiple steps ("...then extend it to handle edge cases")
3. Generate responses for the evolved instructions.
4. Filter for quality and coherence.

This creates a dataset with a natural curriculum of increasing difficulty.

## Why It Matters

### Solving the "Data Wall" Problem

We may be approaching a "data wall" -- the point where most high-quality human-generated text on the internet has already been used for pre-training. Estimates suggest that we will exhaust available high-quality web text within a few years at current training scales. Synthetic data offers a way to continue scaling:

- **Quantity**: An LLM can generate billions of tokens of training data relatively cheaply.
- **Targeting**: Unlike web scraping, synthetic generation can produce data tailored to specific gaps in model capability.
- **Customization**: Data can be generated in specific formats, difficulty levels, and domains on demand.

### Democratizing Model Training

Before synthetic data, creating a high-quality instruction-tuning dataset required either expensive human annotation (thousands of dollars for a few thousand examples) or access to proprietary datasets. Synthetic generation democratized this: the Alpaca recipe showed that anyone with API access to a strong model could create a competitive fine-tuning dataset.

### The Distillation Flywheel

Synthetic data enables a powerful flywheel:
1. A large teacher model generates training data.
2. A smaller student model is trained on this data.
3. The student model, now improved, can generate better data for an even smaller model (or for further self-improvement).

This has led to remarkably capable small models (under 10B parameters) that punch far above their weight.

## Key Technical Details

- **Model collapse**: When models are recursively trained on synthetic data from previous model generations, diversity decreases and errors compound. Shumailov et al. (2023) showed that iterative training on self-generated data causes the model's output distribution to progressively narrow, eventually "collapsing" to a low-diversity mode. Mixing synthetic data with real human data mitigates this.

- **Diversity is the key challenge**: LLMs tend to generate text that clusters around common patterns. Without deliberate diversity-promoting strategies (varying prompts, temperatures, system prompts, and explicit diversity instructions), synthetic datasets can be surprisingly homogeneous.

- **Quality filtering is essential**: Not all synthetic data is good. Common filtering strategies include:
  - Using a stronger model to judge quality
  - Deduplication (synthetic data often contains near-duplicates)
  - Perplexity filtering (removing outputs that are too predictable or too surprising)
  - Task-specific verification (running generated code, checking math answers)

- **License and terms-of-service considerations**: Many model providers' terms of service prohibit using outputs to train competing models. The Alpaca project sparked debate about whether training on GPT-3.5 outputs violates OpenAI's terms.

- **Benchmark contamination**: If the teacher model has been exposed to benchmark test sets during its training, it may "leak" benchmark-specific knowledge into synthetic data, artificially inflating the student model's benchmark scores without genuine capability improvement.

- **Optimal synthetic-to-real data ratios** are actively studied but domain-dependent. A common finding is that mixing 50-80% synthetic data with 20-50% real data outperforms either alone.

## Common Misconceptions

- **"Synthetic data is a free lunch."** It inherits the biases, errors, and limitations of the generating model. A teacher model that hallucinates facts will produce training data containing hallucinated facts.
- **"Models trained on synthetic data are just copies of the teacher."** While there is significant knowledge transfer, the student model develops its own behavior patterns shaped by its architecture, size, and the rest of its training. Synthetic data is one ingredient, not a cloning recipe.
- **"More synthetic data is always better."** There are rapidly diminishing returns. After a certain volume, additional synthetic data provides marginal benefit and can even degrade performance through homogeneity.
- **"Synthetic data can fully replace human data."** Current evidence suggests that some human-generated data is always needed to ground the training process and provide genuine diversity. Purely synthetic training pipelines tend to produce models with subtle but systematic deficiencies.
- **"All synthetic data is the same quality."** The generation strategy (prompt design, teacher model quality, filtering pipeline) matters enormously. Carefully curated synthetic data vastly outperforms naively generated synthetic data.

## Connections to Other Concepts

- **Supervised fine-tuning** is the primary consumer of synthetic instruction-response data. SFT on synthetic data is one of the most common and accessible fine-tuning approaches.
- **Constitutional AI** relies on synthetic data -- the critiques, revisions, and AI preference labels are all forms of synthetic training data.
- **DPO** benefits from synthetic preference pairs, where AI judges rank responses to create training data.
- **Knowledge distillation** is the theoretical framework that explains why training on a teacher model's outputs works -- the teacher's outputs contain "dark knowledge" (soft probability distributions) that is richer than hard labels alone.
- **Chain-of-thought training** uses synthetic reasoning traces generated by strong models to teach weaker models how to reason step by step.

## Further Reading

1. **"Self-Instruct: Aligning Language Models with Self-Generated Instructions" (Wang et al., 2023)** -- The foundational paper that introduced the pipeline for bootstrapping instruction-tuning data from a small seed set using model generation.
2. **"Textbooks Are All You Need" (Gunasekar et al., 2023)** -- Microsoft's Phi-1 paper, demonstrating that synthetic textbook-quality data can produce remarkably capable small models for code generation.
3. **"The Curse of Recursion: Training on Generated Data Makes Models Forget" (Shumailov et al., 2023)** -- The key paper on model collapse, showing the risks of recursively training on synthetic data without real data anchoring.
