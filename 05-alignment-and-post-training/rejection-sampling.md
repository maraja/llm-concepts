# Rejection Sampling in Alignment

**One-Line Summary**: Rejection sampling (Best-of-N) generates $N$ candidate responses from a language model, scores each with a reward model, and selects the highest-scoring output -- providing an implicit KL-constrained policy improvement that captured most of the alignment gains in Llama 2, often matching or exceeding PPO while being far simpler to implement.

**Prerequisites**: RLHF pipeline (reward model, policy optimization), KL divergence as a regularizer, reward modeling and the Bradley-Terry framework, supervised fine-tuning, and basic sampling and probability theory.

## What Is Rejection Sampling?

Reinforcement learning algorithms like PPO and GRPO update model weights to produce better outputs. But there is a much simpler way to improve a model's outputs: just generate many candidates and pick the best one. This is rejection sampling, also called Best-of-N sampling.

The analogy is straightforward: imagine you are writing an important email. You do not send the first draft. You write several versions, reread them, and send the one that sounds best. Rejection sampling does exactly this, with a reward model serving as the editor that picks the winner. Generate $N$ complete responses for a given prompt, score each with the reward model, and return the highest-scoring response.

What makes this seemingly simple technique so powerful is its implicit theoretical properties. Selecting the best of $N$ samples is mathematically equivalent to sampling from a modified distribution that is KL-constrained away from the original policy -- exactly the kind of objective that PPO explicitly optimizes, but achieved here without any gradient updates to the model weights. The KL divergence between the Best-of-N distribution and the original policy is approximately $\log(N) - (N-1)/N$, which means increasing $N$ provides diminishing returns. Going from $N = 1$ to $N = 16$ captures much of the available improvement; going from $N = 16$ to $N = 256$ adds relatively little. This is both a strength (it provides natural regularization against reward hacking) and a limitation (you cannot push arbitrarily far without changing the model itself).

## How It Works

### Best-of-N at Inference Time

The simplest application is inference-time rejection sampling:

1. **Sample**: Generate $N$ independent completions from the policy $\pi_\theta$ for prompt $x$: $\{y_1, y_2, \ldots, y_N\}$.
2. **Score**: Evaluate each completion with the reward model: $\{R(x, y_1), R(x, y_2), \ldots, R(x, y_N)\}$.
3. **Select**: Return $y^* = \arg\max_i R(x, y_i)$.

This requires $N$ forward passes through the policy (which can run in parallel as a single batched generation) and $N$ reward model evaluations per prompt. There are no backward passes, no gradient computations, and no weight updates. The approach is embarrassingly parallel -- all $N$ generations can run simultaneously on available hardware.

### The Implicit KL Constraint

The key theoretical insight, established by Stiennon et al. (2020) and further analyzed by Gao et al. (2022), is that Best-of-N sampling from distribution $p$ is equivalent to sampling from a tilted distribution $q^*$ where the KL divergence from the original distribution is bounded:

$$D_{\text{KL}}(q^* \| p) \approx \log(N) - \frac{N-1}{N}$$

The quality improvement scales sublinearly as $O(\sqrt{\log N})$. This has concrete implications:
- $N = 4$: KL $\approx 0.6$ nats, captures roughly 60% of the maximum achievable improvement
- $N = 16$: KL $\approx 1.8$ nats, captures roughly 80% of the maximum improvement
- $N = 64$: KL $\approx 3.2$ nats, captures roughly 90% of the maximum improvement
- $N = 256$: KL $\approx 4.8$ nats, captures roughly 95% of the maximum improvement

This implicit constraint prevents reward hacking: even with a very large $N$, the selected response is never "too far" from what the model would naturally produce, because you are still selecting from the model's own output distribution. You cannot select a response the model would never generate.

### Iterative Rejection Sampling for Training

The real power of rejection sampling for alignment is using it to generate *training data* for subsequent fine-tuning, creating a loop of iterative improvement. The Llama 2 pipeline made this a core technique:

1. **Sample**: For each prompt in the training set, generate $K$ completions from the current policy (Llama 2 used $K = 70$).
2. **Score and select**: Use the reward model to select the best completion per prompt.
3. **Fine-tune**: Train the policy on the selected completions using standard supervised fine-tuning (cross-entropy loss).
4. **Iterate**: Use the improved policy to sample new completions and repeat the process.

Each iteration effectively distills the reward model's preferences into the policy's weights. The updated model generates better candidates in the next round, which when filtered again produce even better training data. The Llama 2 team found that iterative rejection sampling fine-tuning captured the large majority of alignment improvement, with PPO providing only modest additional gains on top.

### Rejection Sampling + DPO (Llama 3)

Llama 3 combined rejection sampling with DPO in a more sophisticated iterative pipeline:

1. Generate many responses per prompt using the current policy.
2. Score with the reward model and select the best and worst responses as preference pairs $(y_w, y_l)$.
3. Fine-tune with DPO on these on-policy preference pairs.
4. Iterate with the improved model, generating fresh preference pairs each round.

This "online DPO" approach addresses DPO's core weakness -- its off-policy nature. Standard DPO trains on pre-collected preference data that may not reflect the current model's output distribution. By using rejection sampling to generate fresh, on-policy data each iteration, the preference pairs are always relevant to the model's current behavior.

## Why It Matters

1. **Simplicity**: No RL algorithm, no critic model, no PPO hyperparameter tuning. The entire approach is generate-score-select, which can be implemented in a few dozen lines of code on top of standard inference and SFT infrastructure.
2. **Stability**: There is no training instability during the selection phase. The only training that occurs is standard supervised fine-tuning on the selected outputs, which is well-understood and reliably stable.
3. **Effectiveness**: In the Llama 2 ablation studies, rejection sampling fine-tuning captured the large majority of alignment improvement. PPO on top of rejection sampling provided only marginal additional gains, mainly on safety-related benchmarks.
4. **Implicit regularization**: The $\log(N)$ KL bound provides natural protection against reward hacking without requiring an explicit KL penalty coefficient or a frozen reference model during the optimization phase.
5. **Training data generation**: Beyond direct alignment, rejection sampling is widely used as a general technique for generating high-quality synthetic training data -- selecting the best model outputs across large batches for subsequent training on diverse tasks.

## Key Technical Details

- **Llama 2 configuration**: $K = 70$ samples per prompt, iterative rejection sampling fine-tuning over multiple rounds, with the reward model itself being periodically re-trained between iterations to prevent stale reward signals.
- **Compute cost**: $N$ forward passes per prompt at inference time. For $N = 64$ with a 70B model, this is substantial but easily parallelizable. At training time, the cost is amortized across the full training dataset.
- **Reward model quality ceiling**: The selected outputs can only be as good as the reward model's ability to discriminate quality. A poorly calibrated reward model leads to poor selection, and reward model errors compound across iterative rounds.
- **Diminishing returns curve**: The relationship between $N$ and quality improvement is $O(\sqrt{\log N})$. Doubling $N$ from 32 to 64 provides much less marginal improvement than doubling from 2 to 4. This sets a practical upper bound on useful sample counts.
- **Comparison with PPO**: Llama 2 reported that rejection sampling + SFT achieved comparable overall alignment quality to PPO, with PPO showing a small advantage mainly on safety-related benchmarks requiring more precise behavioral shaping.
- **On-policy data freshness**: Iterative rejection sampling keeps the training data on-policy (generated by the current model), which is important for avoiding the distribution shift that degrades off-policy methods like standard DPO.

## Common Misconceptions

- **"Rejection sampling is just a simple trick, not real alignment."** The theoretical connection to KL-constrained optimization shows it is performing the same *type* of optimization as PPO, just through sampling rather than gradient updates. Meta's Llama 2 team validated this empirically, finding it captured most alignment gains.
- **"More samples is always better."** Due to the $O(\sqrt{\log N})$ scaling, returns diminish rapidly. Beyond $N \approx 64-128$, the marginal improvement per additional sample is negligible, and the compute cost grows linearly with $N$.
- **"Rejection sampling can only be used at inference time."** Its most impactful use is for *training data generation* in iterative alignment pipelines, where the upfront sampling cost is amortized over the entire training dataset and its effects persist permanently in the model weights.
- **"PPO makes rejection sampling unnecessary."** The Llama 2 and Llama 3 pipelines both used rejection sampling as a core component alongside (not replaced by) RL algorithms. The two approaches are complementary, addressing different aspects of alignment.

## Connections to Other Concepts

- **RLHF**: Rejection sampling can serve as the policy optimization step in RLHF, replacing or complementing PPO. It uses the same reward model but avoids the RL training loop entirely.
- **GRPO**: GRPO's group sampling is conceptually related -- both sample multiple outputs per prompt. GRPO uses all samples to compute policy gradients; rejection sampling uses only the best sample for supervised fine-tuning.
- **DPO**: When combined with rejection sampling (as in Llama 3), the best and worst sampled outputs form on-policy preference pairs for DPO training, addressing DPO's off-policy limitation.
- **Reward modeling**: The quality of rejection sampling is entirely dependent on the reward model's ability to correctly rank outputs. Reward model quality is the binding constraint on what rejection sampling can achieve.
- **Synthetic data**: Rejection sampling is one of the most common techniques for generating high-quality synthetic training data, used far beyond alignment for general data curation and augmentation.

## Further Reading

1. **"Llama 2: Open Foundation and Fine-Tuned Chat Models" (Touvron et al., 2023, arXiv:2307.09288)** -- Demonstrates iterative rejection sampling fine-tuning as a core alignment technique, with detailed ablations comparing it to PPO and quantifying their relative contributions.
2. **"The Llama 3 Herd of Models" (Dubey et al., 2024, arXiv:2407.21783)** -- Describes the combination of rejection sampling with DPO in an iterative online alignment pipeline, advancing the state of the art in open-weight model alignment.
3. **"Scaling Laws for Reward Model Overoptimization" (Gao et al., 2022, arXiv:2210.10760)** -- Establishes the theoretical scaling relationship between Best-of-N sample size, KL divergence, and reward improvement, providing the mathematical foundation for understanding rejection sampling's implicit regularization.
