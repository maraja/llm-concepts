# Chain-of-Thought Training & Reasoning Models

**One-Line Summary**: Chain-of-thought has evolved from a simple prompting trick into a full training paradigm, where models like OpenAI's o1/o3 and DeepSeek-R1 are explicitly trained to produce extended internal reasoning before answering -- representing a fundamental shift from "System 1" to "System 2" thinking in AI.

**Prerequisites**: Understanding of supervised fine-tuning, RLHF, reward modeling (especially process reward models), basic reinforcement learning concepts, and familiarity with chain-of-thought prompting as an inference-time technique.

## What Is Chain-of-Thought Training?

Chain-of-thought (CoT) prompting showed that adding "Let's think step by step" to a prompt dramatically improves reasoning performance. But this was purely an inference-time trick -- the model wasn't *trained* to reason, it was merely *prompted* to.

Chain-of-thought training takes the next step: what if we train models to *always* produce extended reasoning before answering? What if the reasoning itself becomes a first-class training objective?

The analogy is the difference between asking a student to "show their work" on an exam (prompting) versus spending years teaching them mathematical thinking, proof strategies, and problem decomposition (training). The first produces superficial steps; the second produces genuine reasoning capability.

This shift is often described using Daniel Kahneman's framework: traditional LLMs operate in "System 1" mode -- fast, intuitive, pattern-matching responses. Reasoning models aim for "System 2" -- slow, deliberate, step-by-step analytical thinking. The result is models that spend more tokens (and therefore more computation) on harder problems, effectively learning to allocate compute dynamically.

## How It Works

### The Evolution from Prompting to Training

**Stage 1 -- Chain-of-thought prompting (2022)**: Wei et al. showed that including step-by-step examples in few-shot prompts dramatically improved reasoning. Zero-shot variants like "Let's think step by step" (Kojima et al., 2022) required no examples at all.

**Stage 2 -- Fine-tuning on reasoning traces (2023)**: Models were supervised fine-tuned on datasets containing explicit reasoning chains. The key insight: the quality and style of reasoning traces in the training data directly shape the model's reasoning behavior.

**Stage 3 -- RL-trained reasoning models (2024-2025)**: OpenAI's o1/o3 and DeepSeek-R1 represent a paradigm shift where RL is used to train the model to produce long, detailed reasoning chains that lead to correct answers. The model learns *how to reason* rather than *what reasoning looks like*.

### Reinforcement Learning with Verifiable Rewards (RLVR)

A breakthrough technique for training reasoning models is RLVR, which uses tasks where correctness can be automatically verified. The approach works as follows:

1. **Select tasks with verifiable answers**: Mathematics, coding, formal logic, and other domains where the answer can be checked automatically (e.g., unit tests for code, numerical answers for math).

2. **Let the model generate a full reasoning chain and answer**: The model produces an extended "thinking" trace followed by a final answer.

3. **Assign reward based on answer correctness**:

$$R(x, y) = \begin{cases} +1 & \text{if the final answer is correct} \\ -1 & \text{if the final answer is incorrect} \end{cases}$$

4. **Optimize with RL (typically GRPO or PPO variants)**: The model learns which reasoning strategies lead to correct answers and reinforces them.

The key insight is that you don't need to supervise the *reasoning process* -- you only need to verify the *outcome*. The model discovers effective reasoning strategies on its own through RL, often developing approaches that humans didn't explicitly teach.

DeepSeek-R1's training is illustrative. Their paper shows that applying RL with verifiable rewards causes emergent behaviors:
- The model spontaneously learns to re-examine and verify its own work.
- It develops self-correction patterns ("Wait, let me reconsider...").
- Reasoning chains grow longer as the model learns that more thorough reasoning leads to better outcomes.
- These behaviors emerge from the reward signal alone, not from supervised examples.

### Process Reward Models for Step-Level Feedback

While RLVR provides only outcome-level feedback ("was the final answer right?"), process reward models (PRMs) provide step-level feedback ("was each reasoning step valid?").

For a reasoning chain with steps $s_1, s_2, \ldots, s_T$:

$$R_{\text{PRM}}(x, s_1, \ldots, s_T) = \prod_{t=1}^{T} P(\text{step } s_t \text{ is correct} \mid x, s_1, \ldots, s_{t-1})$$

PRMs can be used in two ways:
- **At training time**: As the reward signal for RL, providing denser feedback than outcome-only rewards.
- **At inference time**: For best-of-N selection or tree search, where the model generates multiple reasoning paths and the PRM selects the most promising one.

### Extended Thinking / Reasoning Modes

Modern reasoning models implement an "extended thinking" mode where the model produces a potentially very long internal reasoning chain (sometimes thousands of tokens) before generating the visible response. Key implementation details:

- **Thinking tokens are generated but may be hidden**: The user sees only the final answer (or a summary), while the full reasoning chain is produced internally.
- **Adaptive compute**: Harder problems naturally elicit longer reasoning chains. The model learns to allocate more thinking to more difficult tasks.
- **Budget control**: Some implementations allow setting a "thinking budget" -- a maximum number of tokens the model can use for reasoning.

### The GRPO Algorithm

Group Relative Policy Optimization (GRPO), used by DeepSeek, provides a simplified RL approach for reasoning model training:

1. For each prompt $x$, sample a group of $G$ responses $\{y_1, \ldots, y_G\}$ from the current policy.
2. Compute rewards $\{R_1, \ldots, R_G\}$ for each response.
3. Normalize rewards within the group: $\hat{A}_i = \frac{R_i - \text{mean}(R)}{\text{std}(R)}$.
4. Use the normalized advantage to update the policy, reinforcing above-average responses and penalizing below-average ones.

$$\mathcal{L}_{\text{GRPO}} = -\mathbb{E} \left[ \frac{1}{G} \sum_{i=1}^{G} \min \left( \frac{\pi_\theta(y_i|x)}{\pi_{\text{old}}(y_i|x)} \hat{A}_i, \text{clip}\left(\frac{\pi_\theta(y_i|x)}{\pi_{\text{old}}(y_i|x)}, 1 \pm \epsilon \right) \hat{A}_i \right) \right]$$

GRPO eliminates the need for a separate value/critic network by using group-relative normalization, reducing memory requirements compared to standard PPO.

## Why It Matters

Chain-of-thought training represents a paradigm shift in how we think about LLM capability scaling. The traditional scaling law story was: "bigger models with more data get better." Reasoning models add a new dimension: **test-time compute scaling**. By spending more computation at inference (longer reasoning chains), you can achieve better results without increasing model size.

This has profound implications:
- **Mathematical and scientific reasoning** becomes accessible to LLMs in ways that were previously impossible. o1 scored at the level of PhD students on competition math problems.
- **The compute trade-off changes.** Instead of only scaling up training, you can scale up inference for harder problems.
- **Self-verification emerges.** Models that reason can check their own work, reducing hallucinations on structured tasks.

## Key Technical Details

- **Reasoning doesn't help everything.** Tasks requiring factual recall, creative writing, or social intelligence may not benefit from extended reasoning and can even degrade (overthinking simple questions).
- **Training instability is common.** RL-based reasoning training can exhibit reward hacking, where the model learns to produce "reasoning-like" text that games the reward without genuine logical progress.
- **Distillation is a practical approach.** DeepSeek-R1 showed that fine-tuning smaller models on reasoning traces from larger reasoning models (distillation) can transfer much of the reasoning capability at a fraction of the compute cost.
- **Verification is key.** RLVR works well because math and code provide automatic verification. Extending this to open-ended domains (where correctness is subjective) remains an open challenge.
- **Token efficiency trade-offs.** Reasoning models use many more tokens per response (often 10-100x), increasing latency and cost. This is acceptable for hard problems but wasteful for simple ones.

## Common Misconceptions

- **"Chain-of-thought prompting and chain-of-thought training are the same thing."** Prompting adds reasoning at inference time without changing the model. Training fundamentally changes the model's weights to internalize reasoning as a default behavior.
- **"Longer reasoning chains are always better."** There is an optimal reasoning length for each problem. Over-reasoning can introduce errors, and models can learn to pad reasoning chains without adding substance.
- **"Reasoning models understand logic."** These models have learned effective reasoning *heuristics* through RL. They make logical errors less often than non-reasoning models, but they are not formal logic engines and can still make systematic mistakes.
- **"You need massive compute to train reasoning models."** DeepSeek-R1 showed that relatively efficient RL training (with GRPO) on top of a strong base model can produce competitive reasoning capabilities, and distillation can bring these to even smaller models.

## Connections to Other Concepts

- **RLHF** provides the foundation. Reasoning models use RL optimization but with verifiable rewards rather than (or in addition to) human preference-based reward models.
- **Reward modeling** -- specifically process reward models -- is central to step-level supervision of reasoning.
- **Supervised fine-tuning** on reasoning traces is a complementary approach to RL-based training and is often used in distillation.
- **Inference optimization** becomes more important when models produce long reasoning chains, motivating techniques like speculative decoding and KV-cache optimization.
- **Synthetic data** -- reasoning models can generate synthetic reasoning traces used to train other models, creating a flywheel of reasoning capability.

## Diagrams and Visualizations

![Chain-of-thought prompting example showing how intermediate reasoning steps lead to correct answers](https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/chain-of-thought-examples.png)
*Source: [Lilian Weng – Prompt Engineering](https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/)*

*Recommended visual: Evolution from chain-of-thought prompting to training with reasoning traces (o1/R1 paradigm) — see [OpenAI Learning to Reason](https://openai.com/index/learning-to-reason-with-llms/)*

## Further Reading

1. **"DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning" (DeepSeek, 2025)** -- A detailed and transparent account of training a frontier reasoning model, including the discovery of emergent reasoning behaviors through RL.
2. **"Let's Verify Step by Step" (Lightman et al., 2023)** -- OpenAI's work on process reward models that provide step-level supervision for mathematical reasoning.
3. **"Chain-of-Thought Prompting Elicits Reasoning in Large Language Models" (Wei et al., 2022)** -- The original paper that demonstrated chain-of-thought as a prompting technique, laying the conceptual groundwork for the training paradigm that followed.
