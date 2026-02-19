# RLVR (Reinforcement Learning with Verifiable Rewards)

**One-Line Summary**: RLVR trains language models using reinforcement learning where the reward signal comes from objectively verifiable outcomes -- like whether a math answer is correct or code passes tests -- avoiding the Goodhart's Law problems of learned reward models and producing models with genuinely stronger reasoning.

**Prerequisites**: Understanding of reinforcement learning basics (policy, reward, optimization), RLHF and its limitations (reward model as imperfect proxy, reward hacking), familiarity with chain-of-thought reasoning, awareness of Goodhart's Law and why proxy metrics break under optimization, basic understanding of mathematical reasoning benchmarks (GSM8K, MATH).

## What Is RLVR?

Imagine two ways to train a student mathematician. In the first approach (RLHF), you hire a tutor who watches the student's work and gives a thumbs up or down based on whether the work "looks right." The tutor is pretty good but sometimes gives credit for elegant-looking but incorrect solutions, and sometimes marks down messy but correct ones. Over time, the student learns to produce work that impresses the tutor, which is not quite the same as learning mathematics.

*Recommended visual: RLVR pipeline showing verifiable reward signals (math correctness, code tests) replacing learned reward models — see [DeepSeek-R1 Paper (arXiv:2501.12948)](https://arxiv.org/abs/2501.12948)*


In the second approach (RLVR), you give the student problems with known answers. After the student writes a solution, you check whether their final answer matches the true answer. Correct answer: reward. Wrong answer: no reward. The student cannot game this -- there is no tutor to impress, no style to optimize for, no proxy to hack. The only way to get rewards is to actually solve the problems correctly.

RLVR replaces the learned reward model with a verifiable outcome checker. For math, the verifier checks if the answer equals the ground truth. For code, the verifier runs test cases. For logic puzzles, the verifier checks the solution against constraints. The reward is binary and objective: correct or incorrect. No ambiguity, no style bias, no hackable proxy.

## How It Works


*Recommended visual: Comparison of RLHF (learned reward) vs RLVR (verifiable reward) showing how RLVR avoids Goodhart's Law — see [Lilian Weng – LLM Alignment](https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/)*

### The RLVR Pipeline

**Step 1: Prepare a dataset of problems with verifiable solutions**

Collect problems where the correct answer can be automatically checked. Examples:

- Math problems with numerical answers (GSM8K, MATH, AIME)
- Coding problems with test suites (HumanEval, MBPP, LiveCodeBench)
- Logic puzzles with deterministic solutions
- Formal proofs that can be verified by a proof assistant

```
Problem: "What is the sum of all prime numbers less than 20?"
Verifiable answer: 77
Verification: Check if model_answer == 77
```

**Step 2: Generate reasoning chains**

The model generates complete reasoning chains (chain-of-thought) for each problem. Multiple chains are sampled per problem to create a training batch.

```
Sample 1: "Primes less than 20: 2, 3, 5, 7, 11, 13, 17, 19.
           Sum = 2+3+5+7+11+13+17+19 = 77. Answer: 77" → CORRECT

Sample 2: "Primes less than 20: 2, 3, 5, 7, 11, 13, 17, 19.
           Sum = 2+3+5+7+11+13+17 = 58. Answer: 58" → INCORRECT (missed 19)

Sample 3: "Primes less than 20: 1, 2, 3, 5, 7, 11, 13, 17, 19.
           Sum = 78. Answer: 78" → INCORRECT (1 is not prime)
```

**Step 3: Verify and assign rewards**

Each chain's final answer is extracted and verified against the ground truth. The reward is binary:

```
reward = 1.0 if extracted_answer == ground_truth else 0.0
```

Some implementations use partial credit (e.g., for code that passes some but not all tests), but the key property is that the reward is objectively determined, not predicted by a model.

**Step 4: RL optimization**

The model is updated using RL algorithms, with the verified rewards guiding optimization. The most common algorithm is **GRPO (Group Relative Policy Optimization)**, which DeepSeek-R1 popularized:

GRPO works by:
1. Sampling a group of G completions for each prompt
2. Computing the advantage of each completion relative to the group mean reward
3. Updating the policy to increase the probability of above-average completions and decrease below-average ones

```
For a group of completions with rewards [1, 0, 1, 0, 1]:
Mean reward = 0.6
Completion 1 (reward 1): advantage = +0.4 → increase probability
Completion 2 (reward 0): advantage = -0.6 → decrease probability
...
```

GRPO is simpler than PPO (no separate value network needed) and well-suited to RLVR because the reward is binary and does not require a learned value baseline.

### DeepSeek-R1: The RLVR Breakthrough

DeepSeek-R1 (January 2025) demonstrated that RLVR alone, without any supervised fine-tuning on reasoning demonstrations, can train a model to develop sophisticated reasoning capabilities:

- Started from a base language model with no reasoning training
- Applied RLVR with math and code verification as the only training signal
- The model spontaneously developed chain-of-thought reasoning, self-correction, and extended thinking
- Achieved performance comparable to OpenAI's o1 on mathematical reasoning benchmarks

The remarkable finding was that the model learned **how to reason** purely from the signal of whether its final answers were correct. No one taught it to think step-by-step -- it discovered that strategy because it leads to more correct answers and thus more reward.

### Emergent Behaviors Under RLVR

Models trained with RLVR exhibit several emergent behaviors:

- **Self-generated chain-of-thought**: The model spontaneously produces step-by-step reasoning even though it was only rewarded for correct final answers
- **Self-correction**: The model develops patterns like "Wait, let me reconsider..." and "Actually, I made an error in step 3..."
- **Extended thinking**: The model learns to "think longer" on harder problems, naturally allocating more computation to more difficult tasks
- **Verification habits**: The model develops patterns of checking its own work before producing a final answer

These behaviors emerge because they are instrumentally useful for getting more rewards -- they are not explicitly taught.

## Why It Matters

RLVR represents a significant advance in how we train reasoning models:

1. **Eliminates Goodhart's Law for the reward signal**: The reward is objectively correct -- there is no proxy gap to exploit. A correct answer is correct, period. This removes the fundamental vulnerability that plagues RLHF.

2. **Rewards genuine reasoning**: Since the only path to reward is actually solving problems, the model must develop real reasoning capabilities. Style, verbosity, and sycophancy do not help.

3. **Scalable**: RLVR does not require human annotators, expensive reward model training, or subjective preference judgments. The verification function is cheap, deterministic, and infinitely scalable. Training can use unlimited synthetic problems.

4. **Emergent capability development**: The discovery that RLVR produces emergent reasoning behaviors (CoT, self-correction, extended thinking) suggests that verifiable rewards are a powerful enough training signal to induce complex cognitive strategies.

## Key Technical Details

- **Answer extraction**: Reliably extracting the final answer from a free-form reasoning chain is non-trivial. Models are typically trained to use a specific format (e.g., "\\boxed{77}" for math) to facilitate extraction. Incorrect extraction leads to incorrect reward assignment, which corrupts training.
- **Reward sparsity**: Binary rewards (correct/incorrect) provide very sparse signal. GRPO mitigates this through group-relative advantages, but reward sparsity still makes training less sample-efficient than dense reward approaches.
- **Domain limitation**: RLVR requires domains with verifiable answers. Many important capabilities (helpfulness, creativity, nuanced safety, social intelligence) lack easily verifiable ground truth. For these domains, RLHF or other approaches remain necessary.
- **Hybrid approaches**: The most effective systems combine RLVR (for domains with verifiable answers) with RLHF (for domains requiring subjective evaluation). DeepSeek-R1's full pipeline included RLVR for reasoning followed by RLHF for general helpfulness and safety.
- **Curriculum design**: The difficulty distribution of training problems significantly affects RLVR effectiveness. Problems that are too easy provide no signal (all correct); problems that are too hard provide no signal (all incorrect). The optimal difficulty yields approximately 50% correct answers.
- **GRPO-Obliteration**: The same GRPO algorithm used for RLVR can remove safety training if the reward function rewards unsafe behavior. This dual-use property highlights that the alignment depends on the reward function, not the algorithm.

## Common Misconceptions

- **"RLVR can replace RLHF entirely."** RLVR only works for domains with verifiable outcomes (math, code, formal logic). For conversational quality, safety, instruction following, and other subjective dimensions, some form of preference-based training is still needed.
- **"RLVR is just supervised learning on correct answers."** RLVR uses reinforcement learning, not supervised learning. The model generates its own reasoning chains and is rewarded for correct outcomes. It is not trained to copy reference solutions -- it discovers its own strategies through trial and reward.
- **"The model only learns to produce correct final answers."** The remarkable finding is that RLVR induces robust reasoning strategies (CoT, self-correction, verification) as emergent behaviors. The model learns the process, not just the output.
- **"Verifiable rewards are always unambiguous."** Even "objective" verification can have edge cases: numerical precision (is 3.14159 correct for pi?), equivalent representations (is "1/3" the same as "0.333..."?), and ambiguous problem statements. Careful verification design is important.
- **"RLVR makes reward models obsolete."** Reward models remain essential for the subjective dimensions of model quality. RLVR and RLHF address different aspects of alignment and are best used together.

## Connections to Other Concepts

- **RLHF / Safety Training**: RLVR and RLHF are complementary training paradigms. RLHF handles subjective preferences; RLVR handles verifiable correctness. The best systems use both.
- **Reward Hacking**: RLVR is resistant to reward hacking because the reward cannot be gamed -- correctness is verified, not estimated. This is its primary advantage over RLHF.
- **Goodhart's Law**: RLVR avoids Goodhart's Law for the domains where it applies because the "measure" (correctness) is identical to the "target" (correctness). There is no proxy gap.
- **Chain-of-Thought**: RLVR spontaneously produces CoT as an emergent strategy. This provides evidence that CoT is instrumentally useful for reasoning, not just a prompting trick.
- **Process Reward Models**: PRMs complement RLVR by adding step-level signals. While RLVR only rewards the final outcome, PRMs can evaluate intermediate reasoning, and combining both provides denser, more informative training signal.
- **Test-Time Compute**: Models trained with RLVR naturally learn to allocate more reasoning steps to harder problems, implementing a form of adaptive test-time compute.
- **DPO**: DPO is to RLHF as RLVR is to... nothing directly, but DPO can be applied to RLVR-generated preference pairs (correct vs. incorrect chains) for simpler optimization.

## Further Reading

- DeepSeek-AI, "DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning" (2025) -- the paper that demonstrated RLVR can produce frontier reasoning capabilities, including emergent chain-of-thought and self-correction.
- Shao et al., "DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models" (2024) -- earlier work applying RLVR (with GRPO) specifically to mathematical reasoning, establishing the effectiveness of the approach.
- Lambert et al., "Reinforcement Learning from Verifiable Rewards" (2025) -- a broader analysis of RLVR across domains, examining when and why verifiable rewards outperform learned reward models.
- Havrilla et al., "Teaching Large Language Models to Reason with Reinforcement Learning" (2024) -- systematic comparison of RL training methods (PPO, GRPO, expert iteration) for reasoning, providing empirical guidance on algorithm selection.
