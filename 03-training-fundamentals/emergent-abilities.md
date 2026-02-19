# Emergent Abilities

**One-Line Summary**: Emergent abilities are capabilities that appear to arise suddenly and unpredictably in large language models once they cross certain scale thresholds -- sparking both excitement about potential breakthroughs and deep concern about our ability to forecast and control AI systems.

**Prerequisites**: Scaling laws (how performance changes with model size), pre-training basics, evaluation benchmarks and metrics, the distinction between training loss and downstream task performance, in-context learning (few-shot prompting).

## What Are Emergent Abilities?

Imagine you are gradually heating water. From 0 to 99 degrees Celsius, nothing dramatic happens -- the water gets warmer, but it remains liquid. Then at 100 degrees, it suddenly boils, undergoing a phase transition to steam. The ability to become steam was not present at 50 degrees, not present at 90 degrees, and then seemingly appeared from nowhere at 100 degrees.

![BIG-Bench task performance curves showing the sharp phase transitions in accuracy as model scale increases, with some tasks jumping from near-random to high accuracy at specific parameter thresholds](https://raw.githubusercontent.com/google/BIG-bench/main/bigbench/benchmark_tasks/results/plot_all_tasks.png)
*Source: [Google BIG-Bench Repository](https://github.com/google/BIG-bench)*


Emergent abilities in LLMs are often described through a similar lens. As models scale from millions to billions to hundreds of billions of parameters, certain capabilities appear to be absent in smaller models and then suddenly present in larger ones. A 1B parameter model cannot do multi-step arithmetic. A 10B parameter model still cannot. Then a 100B+ parameter model seemingly "unlocks" this ability -- not through any architectural change or special training, but purely through increased scale.

The term "emergence" was formalized in the LLM context by Wei et al. (2022), who defined an emergent ability as one that is "not present in smaller models but is present in larger models." This definition, and indeed whether emergence is real at all, has become one of the most debated topics in AI research.

## How It Works


*See the emergent abilities compilation figure in: [Wei et al., "Emergent Abilities of Large Language Models" (arXiv:2206.07682)](https://arxiv.org/abs/2206.07682), Figure 2, which shows performance on multiple tasks across model families (GPT-3, LaMDA, PaLM) with clear phase transitions from random to above-chance accuracy.*

### The Observed Pattern

When plotting task accuracy against model scale (parameters or compute) on standard benchmarks, researchers observed two distinct patterns:

**Pattern 1 -- Smooth improvement (non-emergent)**: Performance improves gradually and predictably with scale. If a 7B model scores 40% and a 13B model scores 50%, you might reasonably predict that a 70B model will score around 65-70%. This is what scaling laws predict.

**Pattern 2 -- Apparent emergence**: Performance is near-random (close to chance level) for all model sizes up to some threshold, then jumps sharply to well-above-chance performance. A 7B model scores 25% (random for 4-choice questions), a 30B model scores 26%, and then a 70B model scores 75%. The capability appears to "emerge" discontinuously.

### Classic Examples of Claimed Emergent Abilities

- **Multi-step arithmetic**: Small models cannot perform three-digit addition; large models can.
- **Chain-of-thought reasoning**: The ability to solve problems by generating intermediate reasoning steps appears only in models above roughly 60-100B parameters.
- **In-context learning**: Few-shot prompting (providing examples in the prompt) is far more effective in larger models.
- **Word unscrambling**: Rearranging scrambled letters into words.
- **International Phonetic Alphabet transliteration**: Converting between IPA and standard text.
- **Instruction following**: Understanding and executing novel instructions not seen during training.

Wei et al. (2022) catalogued over 130 such tasks where performance appeared to emerge discontinuously with scale across multiple model families (GPT-3, LaMDA, PaLM, Chinchilla).

### The Mathematical Perspective

If training loss $L$ scales smoothly as a power law:

$$L(N) = \left(\frac{N_c}{N}\right)^{\alpha}$$

then how can downstream task performance be discontinuous? This apparent contradiction is at the heart of the emergence debate.

One key observation: training loss measures average next-token prediction quality, while task benchmarks measure **binary success or failure** on structured problems. A model's probability of generating the correct multi-step answer might increase smoothly from 0.001 to 0.01 to 0.1 to 0.5 -- but if the evaluation metric is "exact match accuracy," the score stays near zero until the probability crosses some threshold, then jumps.

### The Debate: Is Emergence Real?

The emergence narrative has been powerfully challenged, most notably by Schaeffer et al. (2023) in "Are Emergent Abilities of Large Language Models a Mirage?" Their core argument:

**Emergence is a metric artifact, not a model property.**

The key insight: whether a capability appears "emergent" depends on the **evaluation metric**, not the model. Specifically:

1. **Nonlinear metrics create the illusion of emergence.** Metrics like "exact match accuracy" and "multiple-choice accuracy" are nonlinear, threshold-based functions of the underlying model capability. If the model's per-token probability of correctness increases smoothly but the task requires getting every token right, the accuracy curve will show a sharp jump.

2. **Smooth metrics reveal smooth improvement.** When the same tasks are evaluated with continuous metrics (like log-likelihood of the correct answer, Brier score, or token-level accuracy), the improvement with scale is smooth and predictable -- no emergence.

3. **Changing the metric changes whether emergence appears.** For the exact same models on the exact same tasks, you can make emergence appear or disappear by choosing different evaluation metrics.

This suggests that what we call "emergence" may simply be the interaction between smoothly improving capabilities and threshold-based evaluation metrics -- analogous to how a gradually dimming light appears to "suddenly" turn off when it drops below the threshold of human perception.

### The Counter-Arguments

Defenders of emergence argue:

*See also the metric artifact analysis in: [Schaeffer et al., "Are Emergent Abilities a Mirage?" (arXiv:2304.15004)](https://arxiv.org/abs/2304.15004), Figure 1, which demonstrates how the same underlying smooth improvement appears emergent or smooth depending on whether exact-match or continuous metrics are used.*


- **Practical emergence matters even if mechanistic emergence does not.** If a capability is useless below a threshold (you need 100% of a math solution, not 50%), then the practical "emergence" is real and relevant regardless of whether the underlying probability improved smoothly.
- **Some abilities may genuinely be discontinuous.** Complex reasoning chains might require a minimum capacity to maintain coherent multi-step inference, creating genuine thresholds in capability.
- **The smooth metrics argument has limits.** For tasks requiring genuine compositional reasoning, even smooth metrics may show nonlinear improvements.

The debate remains unresolved, but the field has moved toward a more nuanced view: **the underlying capabilities likely improve smoothly, but the practical utility of those capabilities can appear to emerge suddenly depending on the task structure and evaluation criteria.**

## Why It Matters

### For Capability Development

Understanding emergence (or its absence) directly affects how organizations plan model development. If capabilities emerge unpredictably, you cannot forecast what a model will be able to do until you train it. If capabilities improve smoothly, you can predict them from smaller-scale experiments -- a far more tractable engineering problem.

### For AI Safety

Emergence has profound safety implications:

- **Unpredictable capabilities**: If dangerous capabilities (deception, manipulation, autonomous self-improvement) emerge suddenly at scale, there may be no warning before a model crosses a critical capability threshold.
- **Difficulty of testing**: If a capability is absent in smaller test models, you cannot develop safety measures for it until it appears in the full-scale model -- at which point it may already be deployed.
- **The alignment tax**: If we cannot predict which capabilities will emerge, we may not know what to align the model against until it is too late.

If emergence is largely a metric artifact, the safety picture is somewhat less alarming (capabilities improve smoothly, giving more warning), though the practical threshold effects can still matter.

### For Capability Forecasting

Organizations and governments attempting to forecast AI progress rely heavily on scaling laws. If capabilities emerge unpredictably, forecasting becomes extremely difficult. If they improve smoothly, forecasting is feasible. The resolution of the emergence debate will significantly affect how confident we should be in AI capability predictions.

## Key Technical Details

- **Scale can refer to multiple axes**: Model parameters, training compute, training data, and even inference-time compute (chain-of-thought, sampling strategies). Emergence has been observed along all of these axes.
- **Emergence often appears at specific capabilities**: It is not that the model becomes broadly smarter overnight; specific tasks transition from impossible to possible at different scale thresholds.
- **Few-shot prompting amplifies emergence effects**: Many "emergent" abilities are measured in the few-shot setting, where the model must learn from in-context examples. The interaction between in-context learning ability and task capability creates additional threshold effects.
- **Post-training can shift thresholds**: Fine-tuning and RLHF can "unlock" capabilities at smaller scales that previously required larger models, suggesting that the capability was latent in the weights but not accessible through pre-training-style prompting.
- **Benchmark saturation**: As models improve, benchmarks that once showed emergence are now "saturated" (all large models score near 100%), requiring ever-harder benchmarks to measure progress.

## Common Misconceptions

- **"Emergence means the model gained a new capability from nowhere."** The underlying representations likely improve smoothly. What changes is whether the capability crosses a practical utility threshold.
- **"Emergent abilities prove models are becoming conscious/sentient."** Emergence in the scaling laws sense is a statement about performance on benchmarks, not about subjective experience or consciousness.
- **"If we cannot predict emergence, we should stop scaling."** This conflates unpredictability of specific task thresholds with fundamental uncontrollability. Even with emergence, many safety measures remain effective.
- **"Emergence has been debunked."** The Schaeffer et al. critique is powerful but not universally accepted. The debate has become more nuanced, not resolved.
- **"All capabilities are emergent."** Many capabilities (text fluency, grammar, basic knowledge recall) improve smoothly and predictably with scale. Only certain complex, structured tasks show the threshold pattern.

## Connections to Other Concepts

- **Scaling Laws**: The smooth power-law improvement in loss contrasts with the apparently discontinuous improvement in task performance.
- **Pre-Training**: Emergence is observed as a function of pre-training scale (parameters, data, compute).
- **Chain-of-Thought Prompting**: One of the most prominent examples of an apparently emergent capability.
- **In-Context Learning**: The ability to learn from examples in the prompt, itself considered emergent, interacts with other emergent capabilities.
- **AI Safety and Alignment**: Emergence directly impacts the ability to forecast and prepare for potentially dangerous capabilities.
- **Evaluation and Benchmarks**: The choice of metric fundamentally affects whether emergence is observed.

## Further Reading

- Wei, J., et al. (2022). "Emergent Abilities of Large Language Models" -- The paper that formalized the concept and catalogued over 130 emergent abilities across model families.
- Schaeffer, R., Miranda, B., & Koyejo, S. (2023). "Are Emergent Abilities of Large Language Models a Mirage?" -- The influential counter-argument showing that emergence may be an artifact of nonlinear evaluation metrics.
- Srivastava, A., et al. (2022). "Beyond the Imitation Game: Quantifying and Extrapolating the Capabilities of Language Models" (BIG-bench) -- The large-scale collaborative benchmark that provided much of the empirical data for studying emergence across hundreds of tasks.
