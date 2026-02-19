# GRPO (Group Relative Policy Optimization)

**One-Line Summary**: GRPO is a reinforcement learning algorithm developed by DeepSeek that eliminates the critic (value) model entirely by estimating advantages through group-based relative scoring of multiple sampled outputs -- dramatically reducing memory requirements while achieving stable, effective policy optimization.

**Prerequisites**: Understanding of RLHF and PPO (policy, reward, advantage estimation, clipped objectives), KL divergence as a regularizer, reward modeling, and basic statistics (mean, standard deviation, z-scores).

## What Is GRPO?

Standard PPO in the RLHF pipeline requires four models in memory simultaneously: the policy, the reference model, the reward model, and a critic (value) model that estimates how good each state is. The critic is essential for computing "advantages" -- how much better an action was compared to what was expected. But training this critic is itself unstable, memory-intensive, and adds another source of error.

GRPO asks: what if we could estimate advantages without a critic at all?

The key insight is surprisingly simple. Instead of training a neural network to predict expected rewards, GRPO samples a *group* of outputs for each prompt and uses the group's own statistics as the baseline. Think of it like grading on a curve: instead of having an external judge estimate what a "good" score should be, you simply compare each student's performance against the class average. If a response scored above the group mean, it gets a positive advantage; below the mean, negative. No external critic needed.

This approach draws from a long lineage in RL -- REINFORCE with baselines, self-play, and rejection sampling -- but packages it into a practical, scalable algorithm that proved powerful enough to train DeepSeek-R1, one of the first models to develop emergent chain-of-thought reasoning purely from reinforcement learning.

## How It Works

### Group-Based Advantage Estimation

For each prompt $x$, GRPO samples a group of $G$ outputs $\{y_1, y_2, \ldots, y_G\}$ from the current policy $\pi_\theta$. Each output is scored by the reward model (or rule-based reward function), producing rewards $\{r_1, r_2, \ldots, r_G\}$. The advantage for each output is computed as a z-score:

$$A_i = \frac{r_i - \text{mean}(\{r_1, \ldots, r_G\})}{\text{std}(\{r_1, \ldots, r_G\})}$$

This is the core innovation. No learned value function, no critic network, no temporal difference learning. The group itself serves as the baseline.

A typical group size is $G = 64$, meaning 64 completions are sampled per prompt per training step. The z-score normalization ensures advantages are zero-mean and unit-variance within each group, which stabilizes gradient magnitudes across prompts that may have very different reward scales.

The variance of this estimator decreases as $O(1/G)$. With $G = 64$, the standard error of the mean is about 12.5% of the standard deviation -- sufficiently precise for stable policy gradient updates.

### The GRPO Objective

Like PPO, GRPO uses a clipped surrogate objective to prevent overly large policy updates:

$$\mathcal{L}_{\text{GRPO}}(\theta) = -\mathbb{E} \left[ \frac{1}{G} \sum_{i=1}^{G} \min \left( \frac{\pi_\theta(y_i|x)}{\pi_{\theta_{\text{old}}}(y_i|x)} A_i, \; \text{clip}\left(\frac{\pi_\theta(y_i|x)}{\pi_{\theta_{\text{old}}}(y_i|x)}, 1-\epsilon, 1+\epsilon\right) A_i \right) + \beta \cdot D_{\text{KL}}(\pi_\theta \| \pi_{\text{ref}}) \right]$$

The clipping ratio $\epsilon$ (typically 0.2) prevents the policy from changing too drastically in a single update. The KL divergence penalty against the reference policy $\pi_{\text{ref}}$ prevents the model from drifting too far from its starting point.

### Token-Level vs. Sequence-Level Advantages

In standard PPO for language models, the critic estimates a value at every token position, enabling per-token advantage computation via Generalized Advantage Estimation (GAE).

GRPO takes a different approach: it assigns a single advantage to the entire sequence and applies it uniformly to every token during the policy gradient update. While per-token advantages provide more fine-grained signal, they require a well-calibrated critic -- exactly what GRPO eliminates.

In practice, sequence-level advantages with large group sizes provide sufficient signal. The policy gradient still differentially upweights tokens in high-advantage sequences and downweights tokens in low-advantage sequences. The per-sequence signal gets refined across many training steps.

### Reward Design for GRPO

The choice of reward function is particularly important because group-based advantage estimation requires meaningful variance in rewards across the group. If all outputs receive similar rewards, advantages will be near-zero and learning stalls.

DeepSeek-R1-Zero used carefully designed rule-based rewards:

- **Correctness reward**: Binary 1/0 for math (exact answer match), or partial credit based on solution structure
- **Format compliance**: Reward for placing answers within designated tags (e.g., `<answer>...</answer>`)
- **Length penalty**: Negative reward for excessively long or repetitive outputs

### The Training Loop in Practice

A single GRPO training iteration proceeds as follows:

1. Sample a batch of prompts from the training set.
2. For each prompt, generate $G = 64$ complete responses using the current policy.
3. Score each response using the reward model or rule-based reward function.
4. Compute z-score normalized advantages within each group.
5. Compute the clipped policy gradient loss across all prompt-response pairs.
6. Add the KL divergence penalty against the reference policy.
7. Update policy parameters. Optionally repeat steps 5-7 for multiple epochs on the same batch.
8. Periodically update the reference policy (or keep it fixed throughout training).

## Why It Matters

1. **Memory efficiency**: GRPO requires roughly half the memory of PPO-based RLHF because it eliminates the critic model entirely. For a 70B parameter policy, this saves ~140GB of GPU memory (critic weights plus optimizer states).
2. **Training stability**: Critic networks in PPO are a major source of instability -- poorly calibrated critics, reward scale sensitivity, and interacting training dynamics. GRPO sidesteps all of these.
3. **Emergent reasoning**: DeepSeek-R1-Zero, trained with GRPO using only rule-based rewards, spontaneously developed chain-of-thought reasoning, self-verification ("let me check my work"), and "aha moments" without any supervised demonstrations.
4. **Simplicity**: GRPO requires fewer hyperparameters than PPO with GAE (no GAE lambda, no value function learning rate, no value loss coefficient).
5. **Scalability**: Larger group sizes give better advantage estimates, and the sampling of $G$ outputs per prompt is embarrassingly parallel across GPUs.

## Key Technical Details

- **Group size**: $G = 64$ is standard. Values from 16 to 256 have been explored. Larger groups provide lower-variance estimates but cost more compute.
- **DeepSeekMath results**: 7B model achieved 58.8% on MATH and 88.2% on GSM8K.
- **DeepSeek-R1-Zero**: Pure RL with GRPO, no SFT stage -- reasoning emerged from RL alone with rule-based rewards.
- **KL penalty coefficient $\beta$**: Typically 0.01-0.04. Too low permits reward hacking; too high prevents learning.
- **Multiple PPO epochs**: GRPO can reuse the same sampled group for 2-4 gradient updates before resampling.
- **Reward compatibility**: Works with both learned reward models and hand-crafted reward functions.
- **Sampling temperature**: Higher temperatures during group sampling increase output diversity, providing better advantage estimates.

## Common Misconceptions

- **"GRPO is just REINFORCE with a baseline."** GRPO adds PPO's clipped objective, explicit KL regularization, and z-score normalization. The combination is far more stable than vanilla REINFORCE.
- **"Eliminating the critic must sacrifice learning quality."** In practice, GRPO matches or exceeds PPO performance. The critic in standard PPO introduces its own errors and instabilities.
- **"Sequence-level advantages lose too much information."** For tasks with holistic rewards (correctness, helpfulness), sequence-level advantages are a natural fit.
- **"GRPO only works with rule-based rewards."** It works equally well with learned reward models. The reward source is independent of the advantage estimation method.
- **"Larger group sizes are always better."** Returns diminish beyond $G = 128$; the marginal variance reduction rarely justifies the additional sampling cost.

## Connections to Other Concepts

- **PPO**: GRPO inherits PPO's clipped objective but replaces the critic with group-based advantage estimation.
- **REINFORCE with baseline**: The intellectual ancestor -- using sampled returns rather than a learned critic. GRPO improves on it with clipping and KL regularization.
- **Rejection sampling**: GRPO's group sampling is related, but uses all samples for policy gradients rather than just the best one.
- **RLHF/RLVR**: GRPO is a drop-in replacement for PPO in the RLHF or RLVR pipeline.
- **Chain-of-thought training**: DeepSeek-R1-Zero's emergent reasoning connects GRPO to the study of how reasoning is elicited through RL.

## Diagrams and Visualizations

*Recommended visual: GRPO algorithm diagram showing group sampling, z-score advantage estimation, and clipped policy update — see [DeepSeekMath Paper (arXiv:2402.03300)](https://arxiv.org/abs/2402.03300)*

*Recommended visual: Comparison of PPO (with critic) vs GRPO (critic-free group-based advantage) — see [DeepSeek-R1 Paper (arXiv:2501.12948)](https://arxiv.org/abs/2501.12948)*

## Further Reading

1. **"DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models" (Shao et al., 2024, arXiv:2402.03300)** -- Introduces GRPO and demonstrates its effectiveness for mathematical reasoning.
2. **"DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning" (DeepSeek-AI, 2025, arXiv:2501.12948)** -- Shows GRPO with rule-based rewards producing emergent chain-of-thought reasoning.
3. **"Proximal Policy Optimization Algorithms" (Schulman et al., 2017)** -- Essential background for understanding GRPO's clipped objective and trust-region approach.
