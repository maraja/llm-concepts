# Inference-Time Scaling Laws

**One-Line Summary**: Performance on reasoning tasks improves predictably as you spend more compute at inference time -- through repeated sampling, extended chain-of-thought, tree search, and verifier-guided selection -- enabling smaller models to match larger ones on hard problems.

**Prerequisites**: Scaling laws (Chinchilla), chain-of-thought prompting, reward models / process reward models (PRM), tree search (MCTS), majority voting / self-consistency

## What Is Inference-Time Scaling?

Training-time scaling laws tell us that bigger models trained on more data perform better. Inference-time scaling laws reveal a parallel truth: you can also improve performance by spending more compute when generating answers, not just when training the model. Think of it as the difference between hiring a smarter employee versus giving an existing employee more time to think. Both approaches improve output quality, and sometimes the latter is more cost-effective.

The core insight, demonstrated by Snell et al. (2024), is that a compute-optimal strategy exists at inference time just as at training time. A 7B model equipped with a process reward model (PRM) and best-of-N sampling can match the performance of a 34B model running standard greedy decoding on math reasoning benchmarks -- at comparable total FLOPs. This means the traditional assumption that "bigger model = better performance" breaks down when you account for inference-time compute allocation.

This paradigm shift became dramatically visible with OpenAI's o1 model family, which uses extended internal reasoning chains to achieve remarkable performance on hard problems. On AIME 2024 (a challenging math competition), o1 improved from 12% at low compute to 83% at high compute. The subsequent o3 model achieved 87.5% on ARC-AGI at high compute settings, a benchmark previously considered resistant to LLM approaches. These results demonstrate that inference-time scaling is not merely an academic curiosity but a practical engineering lever that fundamentally changes what problems LLMs can solve.

## How It Works

### Repeated Sampling and Self-Consistency

The simplest form of inference-time scaling is generating multiple independent samples and selecting the best one. Self-consistency (Wang et al., 2023) generates N reasoning chains and takes a majority vote over the final answers:

```
Query: "What is 17 * 24 + 13?"

Sample 1: 17*24=408, 408+13=421  -> Answer: 421
Sample 2: 17*24=408, 408+13=421  -> Answer: 421
Sample 3: 17*24=398, 398+13=411  -> Answer: 411
...
Sample 40: 17*24=408, 408+13=421 -> Answer: 421

Majority vote: 421 (correct)
```

On GSM8K with PaLM 540B, self-consistency with 40 samples improves accuracy by +17.9% over single-sample chain-of-thought. The scaling is logarithmic: doubling the number of samples yields diminishing but reliable gains. Going from 1 to 4 samples provides the largest marginal improvement, while going from 20 to 40 samples provides a smaller but still meaningful boost.

### Verifier-Guided Search (Best-of-N with PRM)

Process Reward Models (PRMs) score each intermediate reasoning step, not just the final answer. Combined with best-of-N sampling, the PRM selects the sample with the highest-quality reasoning chain:

```
Generate N candidate solutions
  -> PRM scores each reasoning step in each candidate
  -> Aggregate step scores into a chain-level score
  -> Select the highest-scoring chain as the final answer
```

PRM-guided best-of-N outperforms majority voting by +5.8% on the MATH benchmark because it can identify correct solutions even when they are in the minority. The verifier catches reasoning errors that would be invisible to majority vote (where a common wrong approach can outvote a rare correct one).

The PRM scoring function assigns a probability of correctness to each step:

```
Step 1: "Let x = 2y + 3"        -> PRM score: 0.95
Step 2: "Substituting: 4y + 6"  -> PRM score: 0.92
Step 3: "Therefore y = 7"       -> PRM score: 0.31  (likely error detected)
```

This step-level granularity allows the verifier to pinpoint where reasoning goes wrong, enabling more sophisticated search strategies like beam search with step-level pruning.

### Extended Chain-of-Thought and Sequential Reasoning

Rather than sampling independently, models can be trained or prompted to engage in longer sequential reasoning. OpenAI's o1 and o3 models exemplify this: they generate extended internal monologues that break complex problems into sub-steps, backtrack when reaching contradictions, and verify intermediate results.

This represents a different scaling regime from parallel sampling. While best-of-N scales by exploring breadth (many independent attempts), extended CoT scales by exploring depth (longer, more thorough reasoning within a single chain). The two approaches have different scaling characteristics:

- **Search against verifier**: Follows approximate power-law scaling. Doubling compute yields consistent but gradually diminishing returns. Performance scales as approximately `accuracy ~ c * log(N)` where N is the number of samples.
- **Sequential reasoning**: Scaling behavior depends heavily on the problem structure and the model's ability to productively use additional reasoning steps. Some problems benefit enormously from longer reasoning; others hit a ceiling quickly.

These two regimes can also be combined: generate multiple extended reasoning chains and use a verifier to select the best one, achieving the benefits of both breadth and depth.

### Difficulty-Dependent Compute Allocation

A critical finding is that inference-time compute benefits are not uniform across problem difficulties:

- **Easy problems**: Additional compute provides minimal benefit. The base model already solves them reliably with greedy decoding. Spending 100x more compute on a trivial question is pure waste.
- **Medium problems**: Maximum gains. These are problems the model can solve but not reliably -- within the model's "zone of proximal development." More samples, better verification, and longer reasoning chains dramatically improve success rates. This is where inference-time scaling provides the best return on investment.
- **Hard problems**: Diminishing returns. Problems beyond the model's fundamental capability ceiling see limited improvement from additional inference compute, though extended reasoning can still help by decomposing hard problems into easier sub-problems.

This suggests that the optimal strategy is adaptive: allocate more inference compute to medium-difficulty queries and less to easy or impossibly hard ones. A difficulty estimator (which could itself be a lightweight model or classifier) determines the compute budget per query.

### Iterative Refinement

A fifth approach generates an initial response, then iteratively critiques and revises it. Each refinement pass spends additional inference compute to improve the output. This approach is particularly effective for open-ended generation tasks (writing, code) where there is no single correct answer to verify against:

```
Draft -> Self-critique -> Revision -> Self-critique -> Final output
```

Each iteration roughly doubles the inference cost but can yield substantial quality improvements, especially when the critique identifies specific, actionable flaws.

## Why It Matters

1. **Democratization of capability**: Smaller, cheaper models can match larger ones on hard tasks by trading inference compute for training compute, making frontier-level performance accessible without frontier-level model sizes.
2. **Adaptive compute allocation**: Systems can dynamically decide how much inference compute to spend per query, spending more on hard questions and less on easy ones -- unlike fixed-cost per-token pricing.
3. **New capability frontier**: o1 and o3 demonstrate that inference-time scaling unlocks capabilities (competition math, formal reasoning, ARC-AGI) that were previously out of reach for any model at standard inference.
4. **Practical cost optimization**: A 7B model with PRM-guided sampling may be cheaper than a 70B model for equivalent performance on reasoning tasks, because the 7B model's per-sample cost is much lower.
5. **Complementary to training scaling**: Inference-time and training-time scaling compose. The optimal system allocates compute across both dimensions based on the deployment requirements.

## Key Technical Details

- Snell et al. (2024): compute-optimal 7B with PRM matches 34B at standard inference on MATH benchmark
- Self-consistency: +17.9% on GSM8K with 40 samples using PaLM 540B (Wang et al., 2023)
- PRM-guided best-of-N: +5.8% over majority vote on MATH with 256 samples (Lightman et al., 2023)
- o1: 12% to 83% on AIME 2024 scaling from low to high inference compute
- o3: 87.5% on ARC-AGI at high compute budget (~$10K+ per task at high settings)
- Scaling law regimes: search against verifier follows approximate power law; sequential reasoning has problem-dependent scaling
- Optimal sample count depends on problem difficulty: 1 sample for easy, 16-64 for medium, 256+ for hard
- PRM training requires step-level correctness labels, which are expensive to collect (Lightman et al. used human annotators for ~800K step labels)
- Inference-time scaling has diminishing returns past a model-specific capability ceiling -- no amount of compute helps if the model fundamentally lacks required knowledge
- Compute-optimal frontier: for a fixed inference budget, there exists an optimal split between model size and number of samples
- Temperature matters: higher temperatures increase diversity across samples, improving self-consistency but potentially reducing individual sample quality

## Common Misconceptions

- **"Inference-time scaling just means running the model multiple times."** Repeated sampling is the simplest form, but the most powerful approaches involve structured search (tree search with backtracking), learned verifiers (PRMs), and extended sequential reasoning (o1-style chains). These are qualitatively different from naive re-sampling.
- **"Bigger models are always better than smaller models with more inference compute."** Snell et al. showed that a compute-optimal 7B model with verification matches a 34B model at standard inference. The right comparison is total FLOPs, not model size alone.
- **"Inference-time scaling works equally well on all problems."** Benefits are strongly difficulty-dependent. Easy problems see no gain, impossibly hard problems see minimal gain, and medium-difficulty problems in the model's "zone of proximal development" see the largest improvements.
- **"o1's performance comes from a secret architecture."** While the full details are proprietary, the publicly demonstrated results are consistent with known techniques: extended chain-of-thought, process reward models, and search -- applied at unprecedented scale and with sophisticated training.
- **"More samples always helps."** Beyond a certain point, additional samples provide negligible improvement because the model has exhausted its solution diversity. The marginal value of each additional sample decreases logarithmically.

## Connections to Other Concepts

- **Process Reward Models (PRMs)**: PRMs are the verifiers that make inference-time search effective. Without reliable step-level scoring, best-of-N sampling reduces to majority voting, which is less sample-efficient.
- **Chain-of-Thought Prompting**: CoT is the foundation that inference-time scaling builds on. Longer, more structured reasoning chains are the mechanism through which additional compute translates to better answers.
- **Self-Consistency**: The simplest inference-time scaling technique -- majority voting over multiple samples -- which serves as both a practical baseline and a conceptual starting point.
- **Model Routing**: Routing can implement difficulty-dependent compute allocation by sending easy queries to small models (low compute) and hard queries to large models with extended reasoning (high compute).
- **Monte Carlo Tree Search (MCTS)**: Tree search with value estimation is a natural framework for structured inference-time compute allocation, explored in approaches like Tree-of-Thought and reasoning-via-planning.
- **Speculative Decoding**: While speculative decoding speeds up individual samples, inference-time scaling is about improving quality by spending more compute -- opposite goals that can coexist in the same system.

## Further Reading

- Snell, C., Lee, J., Xu, K., & Kumar, A. (2024). "Scaling LLM Test-Time Compute Optimally can be More Effective than Scaling Model Parameters." arXiv:2408.03314.
- Lightman, H., Kosaraju, V., Burda, Y., Edwards, H., Baker, B., Lee, T., Leike, J., Schulman, J., Sutskever, I., & Cobbe, K. (2023). "Let's Verify Step by Step." arXiv:2305.20050.
- Wang, X., Wei, J., Schuurmans, D., Le, Q., Chi, E., Narang, S., Chowdhery, A., & Zhou, D. (2023). "Self-Consistency Improves Chain of Thought Reasoning in Language Models." ICLR 2023. arXiv:2203.11171.
- OpenAI. (2024). "Learning to Reason with LLMs." https://openai.com/index/learning-to-reason-with-llms/.
- Yao, S., Yu, D., Zhao, J., Shafran, I., Griffiths, T. L., Cao, Y., & Narasimhan, K. (2023). "Tree of Thoughts: Deliberate Problem Solving with Large Language Models." NeurIPS 2023. arXiv:2305.10601.
