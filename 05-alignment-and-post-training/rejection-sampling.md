# Rejection Sampling in Alignment

**One-Line Summary**: Rejection sampling (Best-of-N) generates $N$ candidate responses from a language model, scores each with a reward model, and selects the highest-scoring output -- providing an implicit KL-constrained policy improvement that captured most of the alignment gains in Llama 2, often matching PPO while being far simpler.

**Prerequisites**: RLHF pipeline (reward model, policy optimization), KL divergence as a regularizer, reward modeling, supervised fine-tuning, and basic sampling theory.

## What Is Rejection Sampling?

RL algorithms like PPO and GRPO update model weights to produce better outputs. But there is a much simpler way to improve outputs: generate many candidates and pick the best one.

*Recommended visual: Best-of-N sampling pipeline showing N responses generated, scored by reward model, and best selected — see [Llama 2 Paper Figure (arXiv:2307.09288)](https://arxiv.org/abs/2307.09288)*


This is rejection sampling, also called Best-of-N. Generate $N$ complete responses, score each with a reward model, return the highest-scoring one.

The analogy: imagine writing an important email. You do not send the first draft. You write several versions, reread them, and send the best. Rejection sampling does exactly this, with a reward model as the editor.

What makes this powerful is its theoretical properties. Selecting the best of $N$ samples is mathematically equivalent to sampling from a KL-constrained distribution -- exactly what PPO optimizes, but without any gradient updates. The KL is approximately $\log(N) - (N-1)/N$, meaning increasing $N$ gives diminishing returns. $N = 16$ captures ~80% of the available improvement; $N = 256$ adds relatively little.

## How It Works


*Recommended visual: Rejection sampling performance curves showing diminishing returns as N increases — see [Llama 2 Technical Report](https://arxiv.org/abs/2307.09288)*

### Best-of-N at Inference Time

1. **Sample**: Generate $N$ independent completions from policy $\pi_\theta$ for prompt $x$.
2. **Score**: Evaluate each with the reward model: $\{R(x, y_1), \ldots, R(x, y_N)\}$.
3. **Select**: Return $y^* = \arg\max_i R(x, y_i)$.

This requires $N$ forward passes (parallelizable) and $N$ reward evaluations. No backward passes, no weight updates.

### The Implicit KL Constraint

The theoretical insight (Stiennon et al., 2020; Gao et al., 2022): Best-of-N from distribution $p$ is equivalent to sampling from a tilted distribution $q^*$ with:

$$D_{\text{KL}}(q^* \| p) \approx \log(N) - \frac{N-1}{N}$$

Quality improvement scales as $O(\sqrt{\log N})$:

- $N = 4$: KL $\approx 0.6$, ~60% of maximum improvement
- $N = 16$: KL $\approx 1.8$, ~80% of maximum improvement
- $N = 64$: KL $\approx 3.2$, ~90% of maximum improvement
- $N = 256$: KL $\approx 4.8$, ~95% of maximum improvement

This implicit constraint prevents reward hacking: even with large $N$, the selected response stays within the model's natural output distribution.

### Iterative Rejection Sampling for Training

The real power is using rejection sampling to generate training data. The Llama 2 pipeline:

1. **Sample**: Generate $K = 70$ completions per prompt.
2. **Score and select**: Use reward model to select the best per prompt.
3. **Fine-tune**: Train on selected completions with standard SFT (cross-entropy loss).
4. **Iterate**: Use the improved model to sample new completions; repeat.

Each iteration distills the reward model's preferences into the policy's weights. Llama 2 found that iterative rejection sampling captured most alignment improvement, with PPO adding only modest additional gains.

### Rejection Sampling + DPO (Llama 3)

Llama 3 combined rejection sampling with DPO:

1. Generate many responses per prompt with the current policy.
2. Score with reward model; select best and worst as preference pairs $(y_w, y_l)$.
3. Fine-tune with DPO on these on-policy preference pairs.
4. Iterate with the improved model.

This "online DPO" addresses DPO's off-policy weakness by generating fresh, on-policy data each iteration.

## Why It Matters

1. **Simplicity**: No RL algorithm, no critic, no PPO hyperparameters. Generate-score-select in a few dozen lines of code.
2. **Stability**: No training instability during selection. Only standard SFT occurs.
3. **Effectiveness**: Llama 2 ablations showed rejection sampling captured the large majority of alignment improvement.
4. **Implicit regularization**: $\log(N)$ KL bound provides natural reward hacking protection.
5. **Training data generation**: Widely used beyond alignment for synthetic data curation.

## Practical Considerations

- **Temperature**: Higher sampling temperatures ($T = 0.7$-$1.0$) increase diversity, making high-quality outliers more likely. Too high produces incoherent outputs.
- **Batched generation**: All $N$ samples can run in a single batched forward pass, efficient on modern GPUs.
- **Reward model calibration**: Only relative rankings matter for selection, not absolute scores -- a weaker requirement than PPO needs.
- **Storage**: $K = 70$ samples across 100K+ prompts produces terabytes of text. Efficient data pipelines are essential.
- **Verifier combination**: For math/code, rejection sampling pairs naturally with execution-based verification.

## Key Technical Details

- **Llama 2**: $K = 70$ samples per prompt, iterative RS fine-tuning, reward model re-trained between iterations.
- **Compute**: $N$ forward passes per prompt. For $N = 64$ at 70B, substantial but easily parallelized.
- **Reward model ceiling**: Selected outputs can only be as good as the reward model's ranking ability.
- **Diminishing returns**: $O(\sqrt{\log N})$ scaling. Doubling $N$ from 32 to 64 provides much less improvement than doubling from 2 to 4.
- **RS vs. PPO**: Llama 2 reported comparable alignment quality, with PPO showing small advantages mainly on safety benchmarks.
- **On-policy freshness**: Iterative sampling keeps data on-policy, avoiding distribution shift that degrades off-policy methods.

## Common Misconceptions

- **"Rejection sampling is not real alignment."** It performs KL-constrained optimization -- the same type as PPO, via sampling rather than gradients. Llama 2 validated this empirically.
- **"More samples is always better."** $O(\sqrt{\log N})$ scaling means returns diminish rapidly beyond $N \approx 64-128$.
- **"Only useful at inference time."** Most impactful use is training data generation, where sampling cost is amortized over the dataset's lifetime.
- **"PPO makes it unnecessary."** Llama 2 and 3 both used rejection sampling alongside RL. The approaches are complementary.

## Connections to Other Concepts

- **RLHF**: Rejection sampling can replace or complement PPO as the policy optimization step.
- **GRPO**: Both sample multiple outputs per prompt. GRPO uses all for policy gradients; RS uses only the best for SFT.
- **DPO**: Best and worst samples form on-policy preference pairs for DPO (as in Llama 3).
- **Reward modeling**: RS quality is entirely dependent on the reward model's ranking ability.
- **Synthetic data**: RS is a common technique for high-quality synthetic data generation beyond alignment.

## Further Reading

1. **"Llama 2: Open Foundation and Fine-Tuned Chat Models" (Touvron et al., 2023, arXiv:2307.09288)** -- Demonstrates iterative rejection sampling as a core alignment technique with ablations comparing to PPO.
2. **"The Llama 3 Herd of Models" (Dubey et al., 2024, arXiv:2407.21783)** -- Combines rejection sampling with DPO in iterative online alignment.
3. **"Scaling Laws for Reward Model Overoptimization" (Gao et al., 2022, arXiv:2210.10760)** -- Establishes the theoretical scaling relationship between Best-of-N, KL divergence, and reward improvement.
