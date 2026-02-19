# Bias & Fairness in LLMs

**One-Line Summary**: LLMs absorb and amplify the biases present in their training data, producing outputs that can systematically disadvantage or misrepresent certain groups -- and fully eliminating this bias may be fundamentally impossible.

**Prerequisites**: Understanding of how LLMs are trained on large web-scraped corpora, basics of statistical distributions and representation, familiarity with RLHF and fine-tuning concepts.

## What Is Bias in LLMs?

Imagine training a model on the entire internet. The internet reflects human society, and human society is not equitable. Historical texts describe women as nurses and men as doctors. News articles disproportionately associate certain ethnic groups with crime. English-language content vastly outnumbers content in most other languages. Cultural perspectives from wealthy Western countries dominate.

An LLM trained on this data does not merely reflect these patterns -- it **absorbs them as statistical regularities** and reproduces them in its outputs. When asked to complete "The nurse walked into the room. She..." the model assigns "she" a higher probability than "he" not because it believes nurses must be women, but because that pattern appeared more frequently in its training data.

This is bias in LLMs: systematic skew in model outputs that reflects and can reinforce societal inequities. Fairness, conversely, is the aspiration that model outputs treat different groups equitably -- though defining exactly what "equitably" means turns out to be surprisingly difficult and context-dependent.

## How It Works

### Sources of Bias

Bias enters the system at multiple stages, and understanding these stages is essential for addressing the problem.

**Training data bias** is the most fundamental source. Web-scraped text overrepresents certain demographics, viewpoints, and cultural contexts. Common Crawl and similar corpora contain disproportionately English-language, Western, and male-authored content. Historical texts encode historical prejudices. Reddit and social media content skews toward particular demographics and discourse patterns. Even data that appears balanced may contain subtle correlational biases (e.g., names correlated with race appearing alongside different adjectives).

**Annotation and labeling bias** enters during supervised fine-tuning and RLHF. Human annotators bring their own cultural perspectives, implicit biases, and varying interpretations of guidelines. Studies have shown that annotator demographics significantly affect labeling decisions, especially for subjective tasks like toxicity detection and sentiment analysis. RLHF reward models trained on annotator preferences encode annotator biases into the model's optimization objective.

**Algorithmic and architectural bias** emerges from modeling choices. Tokenization can disadvantage languages with different morphological structures (many non-English languages require more tokens per word, increasing cost and degrading performance). Attention mechanisms may weight more frequent patterns more heavily. The optimization objective itself (next-token prediction) treats frequency as a proxy for importance.

**Deployment and feedback bias** occurs after the model is in production. If biased outputs lead to biased user interactions, and those interactions are used for further fine-tuning, a feedback loop is created that can amplify initial biases over time.

### Types of Bias

- **Gender bias**: Associating professions, traits, and behaviors with specific genders. Models consistently associate STEM fields with male pronouns and caregiving roles with female pronouns at rates that exceed even real-world disparities.

- **Racial and ethnic bias**: Differential treatment based on names, dialects, or cultural contexts associated with different racial or ethnic groups. Studies have shown that models generate more negative sentiment when prompted with names stereotypically associated with certain racial groups.

- **Cultural and geographic bias**: Defaulting to Western (particularly American) cultural norms, holidays, foods, naming conventions, and value systems. When asked about "the president," models default to the US president. When asked about "normal" family structures, they default to Western nuclear family models.

- **Representation bias**: Some groups are simply underrepresented in training data, leading to worse model performance and less nuanced understanding. Indigenous languages, minority cultural practices, and non-Western philosophical traditions receive markedly less sophisticated treatment.

- **Intersectional bias**: Biases that emerge at the intersection of multiple identity categories (e.g., Black women, elderly disabled individuals) can be worse than the sum of individual category biases and are harder to detect with single-axis evaluation.

### Measurement and Benchmarks

Several benchmarks have been developed to measure LLM bias systematically:

- **BBQ (Bias Benchmark for QA)**: Tests for social biases across nine categories (age, disability, gender, nationality, physical appearance, race/ethnicity, religion, socioeconomic status, sexual orientation) using ambiguous and disambiguated question-answering scenarios. Measures whether the model defaults to stereotypical assumptions when evidence is ambiguous.

- **WinoBias**: Extends the Winograd Schema Challenge to test gender bias in coreference resolution. Provides pairs of sentences where gender-stereotypical associations should not affect which entity a pronoun refers to.

- **StereoSet**: Measures stereotypical bias across gender, profession, race, and religion by presenting the model with pairs of stereotypical and anti-stereotypical completions and measuring preference. Also measures language modeling ability, capturing the trade-off between debiasing and general capability.

- **CrowS-Pairs**: Provides paired sentences measuring bias across nine categories, where one sentence is stereotyping and the other is anti-stereotyping. Measures whether the model assigns higher probability to stereotypical completions.

- **RealToxicityPrompts**: While focused on toxicity rather than bias per se, reveals differential toxicity rates across demographic contexts.

## Why It Matters

Biased LLM outputs have real-world consequences that scale with adoption:

**Employment**: LLMs used in resume screening, job description generation, or interview evaluation can systematically disadvantage candidates from certain groups. A model that associates "leadership" with male-coded language may rank qualified women lower.

**Healthcare**: Models providing medical information may give less accurate or less detailed information about conditions that disproportionately affect underrepresented groups, or may encode historical biases in medical research.

**Education**: LLM-powered tutoring systems that perform worse for students using non-standard English dialects effectively provide unequal educational support along socioeconomic and racial lines.

**Legal and criminal justice**: Models used for legal research or risk assessment may encode biases from historical legal texts that reflect discriminatory practices.

**Content creation and media**: LLMs generating stories, images, or marketing content may default to stereotypical representations, reinforcing rather than challenging societal biases.

The scale of LLM deployment means that even small biases, applied billions of times across millions of users, can have massive aggregate impact on societal equity.

## Key Technical Details

- Bias exists on a spectrum from **allocational** (distributing resources or opportunities unfairly) to **representational** (reinforcing stereotypes or rendering groups invisible). Both matter, but allocational harms are often more immediately measurable.
- **Debiasing techniques** include data-level interventions (balancing training data, counterfactual data augmentation), training-level interventions (modifying the loss function, adversarial debiasing), and post-hoc interventions (output filtering, prompt engineering, calibration adjustments).
- **Counterfactual data augmentation** generates training examples where demographic attributes are swapped (e.g., replacing "he" with "she" in professional contexts) to reduce correlational biases. This helps with simple associations but struggles with intersectional and contextual biases.
- **Prompt engineering for fairness** (e.g., instructing the model to "consider diverse perspectives" or "avoid assumptions based on names or demographics") provides modest improvement at low cost but is not robust.
- RLHF can be used to specifically penalize biased outputs, but this requires annotators to reliably identify bias, which itself introduces annotation bias.
- **Perfect debiasing is likely impossible** for at least two reasons: (1) there is no universal agreement on what unbiased output looks like (different fairness definitions are mathematically incompatible), and (2) removing all demographic correlations would strip the model of genuine knowledge about how demographic factors interact with real-world outcomes.
- The **safety-capability trade-off**: aggressive debiasing can reduce model utility. A model trained to never associate gender with any profession cannot accurately answer "What was the gender distribution of US nurses in 1950?"

## Common Misconceptions

- **"Bias in LLMs is a training data problem that can be solved with better data."** While training data is the primary source, bias also enters through annotation, architecture, and optimization. Even with perfectly balanced data, the model's statistical learning process can introduce new biases.
- **"Debiasing means treating everyone the same."** Fairness often requires differential treatment. A medical model should note that sickle cell disease disproportionately affects certain populations -- ignoring this in the name of "equal treatment" would produce worse health outcomes.
- **"We can measure bias with a single metric."** Different fairness metrics (demographic parity, equalized odds, calibration, individual fairness) are mathematically incompatible in most cases. No single number captures whether a model is "fair."
- **"Bias is a binary: models are either biased or not."** All models have some degree of bias. The question is the magnitude, the direction, the context, and whether it causes harm.
- **"Human annotators can reliably identify bias."** Research shows substantial disagreement among annotators about what constitutes biased output, and annotator judgments are themselves influenced by their demographic backgrounds and cultural contexts.

## Connections to Other Concepts

- **RLHF and Alignment**: RLHF is both a tool for reducing bias (by training on less biased human preferences) and a source of bias (annotator preferences encode annotator biases).
- **The Alignment Problem**: Bias is a specific instance of misalignment -- the model's statistical patterns diverge from what we actually want (equitable treatment).
- **Red Teaming**: Bias testing is a core component of red teaming exercises, with specialized teams probing for demographic biases across many dimensions.
- **Guardrails**: Content filters can catch some egregiously biased outputs, but subtle, systemic bias typically passes through output filters undetected.
- **Hallucination**: The model's tendency to confabulate can interact with bias -- it may hallucinate "facts" that align with stereotypical patterns absorbed from training data.

## Diagrams and Visualizations

*Recommended visual: Sources of bias in the LLM pipeline: training data, annotation, model architecture, and deployment — see [Hugging Face Ethics Documentation](https://huggingface.co/docs/hub/model-cards)*

*Recommended visual: Gender bias in word embeddings showing stereotypical associations (he:doctor :: she:nurse) — see [Bolukbasi et al. Debiasing Paper (arXiv:1607.06520)](https://arxiv.org/abs/1607.06520)*

## Further Reading

- Gallegos et al., "Bias and Fairness in Large Language Models: A Survey" (2024) -- Comprehensive survey covering bias sources, measurement approaches, and mitigation strategies across the LLM lifecycle.
- Parrish et al., "BBQ: A Hand-Built Bias Benchmark for Question Answering" (2022) -- Introduces the BBQ benchmark with careful attention to ambiguous vs. disambiguated contexts for measuring social bias.
- Blodgett et al., "Language (Technology) is Power: A Critical Survey of Bias in NLP" (2020) -- Foundational critical analysis of how bias is defined, measured, and discussed in NLP research, arguing for more precise and socially grounded frameworks.
