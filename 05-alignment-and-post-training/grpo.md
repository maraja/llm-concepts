# GRPO (Group Relative Policy Optimization)

**One-Line Summary**: GRPO is a reinforcement learning algorithm developed by DeepSeek that eliminates the critic (value) model entirely by estimating advantages through group-based relative scoring of multiple sampled outputs -- dramatically reducing memory requirements while achieving stable, effective policy optimization.

**Prerequisites**: Understanding of RLHF and PPO (policy, reward, advantage estimation, clipped objectives), KL divergence as a regularizer, reward modeling and the Bradley-Terry preference model, and basic statistics (mean, standard deviation, z-scores).

## What Is GRPO?

Standard PPO in the RLHF pipeline requires four models in memory simultaneously: the policy, the reference model, the reward model, and a critic (value) model that estimates how good each state is. The critic is essential for computing "advantages" -- how much better an action was compared to what was expected. But training this critic is itself unstable, memory-intensive, and adds another source of error. GRPO asks: what if we could estimate advantages without a critic at all?

The key insight is surprisingly simple. Instead of training a neural network to predict expected rewards, GRPO samples a *group* of outputs for each prompt and uses the group's own statistics as the baseline. Think of it like grading on a curve: instead of having an external judge estimate what a "good" score should be, you simply compare each student's performance against the class average. If a response scored above the group mean, it gets a positive advantage; below the mean, negative. No external critic needed.

This approach draws from a long lineage in RL -- REINFORCE with baselines, self-play, and rejection sampling -- but packages it into a practical, scalable algorithm that proved powerful enough to train DeepSeek-R1, one of the first models to develop emergent chain-of-thought reasoning purely from reinforcement learning.

## How It Works

### Group-Based Advantage Estimation

For each prompt $x$, GRPO samples a group of $G$ outputs $\{y_1, y_2, \ldots, y_G\}$ from the current policy $\pi_\theta$. Each output is scored by the reward model (or rule-based reward function), producing rewards $\{r_1, r_2, \ldots, r_G\}$. The advantage for each output is then computed as a z-score normalization:

$$A_i = \frac{r_i - \text{mean}(\{r_1, \ldots, r_G\})}{\text{std}(\{r_1, \ldots, r_G\})}$$

This is the core innovation of GRPO. No learned value function, no critic network, no temporal difference learning. The group itself serves as the baseline. A typical group size is $G = 64$, meaning 64 completions are sampled per prompt per training step. The z-score normalization ensures advantages are zero-mean and unit-variance within each group, which stabilizes gradient magnitudes across prompts that may have very different reward scales -- a prompt about simple arithmetic will have different reward distributions than a prompt about complex calculus, but the normalized advantages are comparable.

The variance of this estimator decreases as $O(1/G)$. With $G = 64$, the standard error of the mean is about 12.5% of the standard deviation -- sufficiently precise for stable policy gradient updates without the overhead of a learned critic.

### The GRPO Objective

Like PPO, GRPO uses a clipped surrogate objective to prevent overly large policy updates. For each output $y_i$ in the group:

$$\mathcal{L}_{\text{GRPO}}(\theta) = -\mathbb{E} \left[ \frac{1}{G} \sum_{i=1}^{G} \min \left( \frac{\pi_\theta(y_i|x)}{\pi_{\theta_{\text{old}}}(y_i|x)} A_i, \; \text{clip}\left(\frac{\pi_\theta(y_i|x)}{\pi_{\theta_{\text{old}}}(y_i|x)}, 1-\epsilon, 1+\epsilon\right) A_i \right) + \beta \cdot D_{\text{KL}}(\pi_\theta \| \pi_{\text{ref}}) \right]$$

The clipping ratio $\epsilon$ (typically 0.2) prevents the policy from changing too drastically in a single update. The KL divergence penalty against the reference policy $\pi_{\text{ref}}$ prevents the model from drifting too far from its starting point. The key difference from standard PPO lies in how $A_i$ is computed -- group statistics rather than a critic network.

### Token-Level vs. Sequence-Level Advantages

In standard PPO for language models, the critic estimates a value at every token position, enabling per-token advantage computation via Generalized Advantage Estimation (GAE). GRPO assigns a single advantage to the entire sequence and applies it uniformly to every token in that sequence during the policy gradient update. This is a meaningful simplification.

While per-token advantages could theoretically provide more fine-grained learning signal (identifying which specific tokens contributed to a high or low reward), they require a well-calibrated critic -- which is exactly what GRPO eliminates. In practice, sequence-level advantages with large group sizes provide sufficient signal for effective learning, because the policy gradient still differentially upweights tokens in high-advantage sequences and downweights tokens in low-advantage sequences. The per-sequence signal gets refined across many training steps as the model learns which generation patterns lead to higher rewards.

### The Training Loop in Practice

A single GRPO training iteration proceeds as follows:

1. Sample a batch of prompts from the training set.
2. For each prompt, generate $G = 64$ complete responses using the current policy.
3. Score each response using the reward model or rule-based reward function.
4. Compute z-score normalized advantages within each group.
5. Compute the clipped policy gradient loss across all prompt-response pairs.
6. Add the KL divergence penalty against the reference policy.
7. Update policy parameters. Optionally repeat steps 5-7 for multiple PPO epochs on the same batch.
8. Periodically update the reference policy (or keep it fixed).

## Why It Matters

1. **Memory efficiency**: GRPO requires roughly half the memory of PPO-based RLHF because it eliminates the critic model entirely. For a 70B parameter policy, this means saving ~140GB of GPU memory (the critic model weights plus its optimizer states).
2. **Training stability**: Critic networks in PPO are a major source of instability -- they can be poorly calibrated, suffer from reward scale issues, and introduce their own training dynamics. GRPO sidesteps all of these problems by using a non-parametric baseline.
3. **Emergent reasoning**: DeepSeek-R1-Zero, trained with GRPO using only rule-based rewards (correctness checks, format compliance), spontaneously developed chain-of-thought reasoning, self-verification ("let me check my work"), and exploration behavior ("aha moments") without any supervised demonstrations of these behaviors.
4. **Simplicity of implementation**: GRPO is conceptually and practically simpler than PPO with GAE, requiring fewer hyperparameters and less engineering effort to get right.
5. **Scalability**: The algorithm scales naturally with compute -- larger group sizes give better advantage estimates, and the sampling is embarrassingly parallel across GPUs.

## Key Technical Details

- **Group size $G = 64$** is the standard in DeepSeek papers, though values from 16 to 256 have been explored. Larger groups provide lower-variance advantage estimates but cost more compute per prompt.
- **DeepSeekMath results**: A 7B model achieved 58.8% on MATH and 88.2% on GSM8K, competitive with significantly larger models and demonstrating GRPO's effectiveness for mathematical reasoning.
- **DeepSeek-R1-Zero** used pure RL with GRPO and no SFT stage -- demonstrating that reasoning can emerge from RL alone with only rule-based reward signals and no human demonstrations.
- **Rule-based rewards** in R1-Zero included: correctness verification (exact match for math), format adherence (placing answers in designated tags), and length penalties to prevent degenerate outputs like excessive repetition.
- **The KL penalty coefficient $\beta$** is critical -- too low permits reward hacking and degenerate behavior, too high prevents meaningful learning. DeepSeek typically uses values around 0.01-0.04.
- **Multiple PPO epochs**: GRPO can reuse the same sampled group for several gradient updates before resampling, improving sample efficiency. Typically 2-4 epochs per group are used.
- **Reward model vs. rule-based**: GRPO works with both learned reward models and hand-crafted reward functions. DeepSeek-R1-Zero used purely rule-based rewards; DeepSeekMath used a combination.

## Common Misconceptions

- **"GRPO is just REINFORCE with a baseline."** While GRPO shares the principle of using sampled returns as baselines, it adds PPO's clipped objective for trust-region optimization, explicit KL divergence regularization against a reference policy, and specific z-score normalization for cross-prompt stability. The combination makes it far more stable than vanilla REINFORCE, which is notoriously high-variance.
- **"Eliminating the critic must sacrifice learning quality."** In practice, GRPO matches or exceeds PPO performance for language model alignment. The critic in standard PPO is itself an approximation that introduces its own errors and instabilities; GRPO's group-based estimation can be more reliable with sufficient group size.
- **"Sequence-level advantages lose too much information."** For language generation tasks where the reward is holistic (was the answer correct? was it helpful?), sequence-level advantages are a natural fit. Token-level advantages are more important in settings where specific tokens are pivotal, such as tool-use or structured output generation.
- **"GRPO only works with rule-based rewards."** While DeepSeek-R1-Zero used rule-based rewards, GRPO works equally well with learned reward models. The choice of reward source is independent of the advantage estimation method.

## Connections to Other Concepts

- **PPO (Proximal Policy Optimization)**: GRPO inherits PPO's clipped objective but replaces the critic with group-based advantage estimation, representing a significant simplification of the RLHF pipeline.
- **REINFORCE with baseline**: The intellectual ancestor of GRPO's approach -- using sampled returns rather than a learned critic to estimate advantages. GRPO improves on REINFORCE with clipping and KL regularization.
- **Rejection sampling**: GRPO's group sampling is conceptually similar to best-of-N rejection sampling, but instead of just keeping the best output, it uses all outputs to compute relative advantages for policy gradient updates.
- **RLHF/RLVR**: GRPO is a drop-in replacement for the PPO stage of RLHF or RLVR, maintaining the same overall pipeline while improving the RL optimization step.
- **Chain-of-thought training**: The emergent reasoning behavior in DeepSeek-R1-Zero connects GRPO to the broader study of how reasoning capabilities can be elicited and trained through reinforcement learning.

## Further Reading

1. **"DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models" (Shao et al., 2024, arXiv:2402.03300)** -- Introduces GRPO and demonstrates its effectiveness for mathematical reasoning, establishing the algorithm's theoretical foundation and practical design choices.
2. **"DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning" (DeepSeek-AI, 2025, arXiv:2501.12948)** -- Shows that GRPO with rule-based rewards can produce emergent chain-of-thought reasoning without any supervised reasoning demonstrations, one of the most striking results in LLM training.
3. **"Proximal Policy Optimization Algorithms" (Schulman et al., 2017)** -- The PPO paper provides essential background for understanding GRPO's clipped objective and trust-region approach to policy updates.
