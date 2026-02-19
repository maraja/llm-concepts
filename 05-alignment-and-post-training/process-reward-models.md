# Process Reward Models (PRMs) vs. Outcome Reward Models (ORMs)

**One-Line Summary**: Process reward models evaluate each intermediate reasoning step for correctness, while outcome reward models only evaluate the final answer -- a distinction that fundamentally changes how AI systems learn to reason, moving from "did you get the right answer?" to "did you reason correctly?"

**Prerequisites**: Understanding of reward models in RLHF (how human preferences are distilled into a model that scores outputs), familiarity with chain-of-thought reasoning and why step-by-step reasoning improves LLM performance, basic understanding of reinforcement learning (reward signals and credit assignment), awareness of reward hacking.

## What Are Process and Outcome Reward Models?

Imagine two math teachers grading student work. Teacher A (the outcome reward model) only looks at the final answer: correct gets full marks, incorrect gets zero. Teacher B (the process reward model) reads every line of work, giving credit for each correct step and flagging where the reasoning goes wrong.

Teacher A's students quickly learn that getting the right answer is all that matters -- some develop tricks and shortcuts that happen to work but reflect no understanding, and others get the right answer by coincidence (two errors canceling out). Teacher B's students learn that the reasoning process matters. They develop robust problem-solving skills because every step must be justified, and they cannot get credit for right answers arrived at through wrong reasoning.

This analogy captures the core distinction:

- **Outcome Reward Models (ORMs)**: Score only the final output. The reward signal is sparse -- one number for the entire generation.
- **Process Reward Models (PRMs)**: Score each intermediate step. The reward signal is dense -- one number per reasoning step.

## How It Works

### Outcome Reward Models (ORMs)

ORMs assign a single scalar score to a complete model output:

```
Input:  "What is 15% of 80?"
Output: "15% means 15/100. So 15/100 * 80 = 12. The answer is 12."
ORM score: 0.95 (high confidence this is correct)
```

The ORM is trained on (input, complete_output, label) pairs, where labels come from human preferences or verified correctness. During RL training, the model receives this single score after generating the full response.

**Strengths**: Simple to train (only need final answer labels), straightforward to apply (one score per output), well-established training pipelines.

**Weaknesses**: Cannot distinguish correct reasoning from lucky guessing. A model that produces "15/100 * 80 = 12" by pattern matching scores the same as one that genuinely understands percentages. The credit assignment problem is severe: if the final answer is wrong, which step caused the error?

### Process Reward Models (PRMs)

PRMs assign a score to each intermediate reasoning step:

```
Input:  "What is 15% of 80?"
Step 1: "15% means 15/100"          → PRM score: 0.98 (correct)
Step 2: "So 15/100 * 80"            → PRM score: 0.97 (correct setup)
Step 3: "= 12"                      → PRM score: 0.99 (correct computation)
Final:  "The answer is 12"          → PRM score: 0.99 (correct)
```

Now consider an incorrect chain:
```
Input:  "What is 15% of 80?"
Step 1: "15% means 15/10"           → PRM score: 0.05 (WRONG: should be 15/100)
Step 2: "So 15/10 * 80 = 120"      → PRM score: 0.90 (computation correct given step 1)
Step 3: "The answer is 120"         → PRM score: 0.10 (wrong final answer)
```

The PRM pinpoints exactly where the reasoning went wrong (step 1), even though step 2's computation was correct given the flawed premise.

### Training PRMs

Training PRMs requires step-level labels, which are more expensive to obtain than outcome labels. Three approaches:

**1. Human Step-Level Annotation**

Human annotators label each reasoning step as correct or incorrect. This is the gold standard but extremely expensive. OpenAI's PRM800K dataset contains 800,000 step-level labels for mathematical reasoning chains, annotated by human labelers.

```
Annotator views each step and marks:
[+] correct step
[-] incorrect step
[?] ambiguous / neutral step
```

**2. Automated Step-Level Labels (Monte Carlo Estimation)**

For domains with verifiable final answers (math, code), you can estimate step correctness automatically:

- At each intermediate step, generate many completions from that point forward
- Check what fraction of completions reach the correct final answer
- A step from which most completions succeed is likely correct
- A step from which most completions fail is likely incorrect

```
Step k correctness ≈ P(correct final answer | steps 1..k)

Estimated by: (# completions from step k that reach correct answer) / (total completions from step k)
```

This is called the "Monte Carlo value estimation" approach and is much cheaper than human annotation, though noisier.

**3. LLM-as-Step-Judge**

Use a strong LLM to evaluate each step's correctness. This is cheaper than human annotation and more scalable, but inherits the biases and limitations of the judge model.

### Using PRMs at Inference Time

PRMs enable powerful test-time compute strategies:

**Best-of-N with PRM scoring**: Generate N complete reasoning chains and select the one with the highest minimum PRM step score (the chain whose weakest step is strongest). This is more effective than ORM-based best-of-N because the PRM can distinguish "correct reasoning with a confident answer" from "lucky guess with shaky reasoning."

**Step-level beam search**: At each reasoning step, generate multiple candidates, score them with the PRM, and keep only the top-k. This prunes incorrect reasoning early, before errors compound into wrong final answers.

**Tree-of-Thought with PRM evaluation**: Use PRM scores as the evaluation function for ToT search, providing more reliable branch pruning than LLM self-evaluation.

## Why It Matters

The PRM vs. ORM distinction addresses some of the deepest challenges in AI reasoning:

1. **Credit assignment**: When a reasoning chain produces a wrong answer, the ORM can only say "this output is bad." The PRM can say "step 3 is where it went wrong." This denser signal makes RL training more effective and helps models learn from their mistakes at the right granularity.

2. **Reward hacking resistance**: It is much harder to reward-hack a PRM than an ORM. An ORM can be fooled by outputs that look right but reason wrongly. A PRM checks the reasoning at each step, making it harder to produce high-scoring outputs through shortcuts.

3. **Interpretable verification**: PRM scores provide a step-by-step audit trail. If a model's reasoning produces an unexpected answer, you can examine which steps the PRM flagged as uncertain, enabling human review at the right level of detail.

4. **Enabling test-time compute**: PRMs are a key ingredient in test-time compute strategies. Without reliable intermediate evaluation, methods like beam search and Tree-of-Thought cannot effectively prune the search space.

## Key Technical Details

- **PRM annotation cost**: Step-level annotation is 3-10x more expensive per example than outcome annotation. The PRM800K dataset required significant investment from OpenAI, and few organizations have produced comparable datasets.
- **Step boundary detection**: Defining where one "step" ends and another begins is non-trivial for free-form text. Mathematical reasoning has natural step boundaries (each equation), but other domains (legal reasoning, creative writing) have less clear boundaries.
- **PRM generalization**: PRMs trained on mathematical reasoning may not generalize to other reasoning domains. Domain-specific PRMs are often needed, increasing the total annotation cost.
- **PRM + ORM combination**: In practice, the best results often come from combining PRM scores (step-level) with ORM scores (output-level). The PRM catches reasoning errors; the ORM catches errors in the final answer format or framing.
- **Self-taught PRMs**: Recent research explores training PRMs from the model's own completions (using Monte Carlo value estimation) rather than human labels, dramatically reducing cost. DeepSeek-R1 used this approach.

## Common Misconceptions

- **"PRMs are just more detailed ORMs."** PRMs represent a fundamentally different training signal: process vs. outcome. A PRM rewards correct reasoning even if the final answer is wrong (due to a later error), and penalizes incorrect reasoning even if the final answer is right (due to lucky error cancellation). This changes what the model learns.
- **"You always need human annotations for PRMs."** Monte Carlo estimation and LLM-as-judge approaches can generate step-level labels automatically for domains with verifiable outcomes. Human annotation is the gold standard but not the only option.
- **"PRMs solve the alignment problem."** PRMs improve reasoning evaluation but only work for domains where step correctness is well-defined. For nuanced domains (ethics, creativity, social interaction), defining what constitutes a "correct step" is itself an unsolved problem.
- **"PRMs are only useful during training."** PRMs are equally valuable at inference time, where they enable best-of-N selection, beam search, and tree search strategies that significantly improve accuracy.

## Connections to Other Concepts

- **Reward Hacking**: PRMs are more resistant to reward hacking because they evaluate the reasoning process, not just the outcome. It is harder to fake correct reasoning at every step than to produce a correct-looking final answer.
- **Goodhart's Law**: PRMs partially mitigate Goodhart's Law by providing a denser, more faithful reward signal. The gap between "high PRM score" and "actually correct reasoning" is smaller than the gap between "high ORM score" and "actually correct."
- **Chain-of-Thought**: CoT produces the reasoning chains that PRMs evaluate. PRMs incentivize models to produce genuinely useful reasoning steps rather than post-hoc rationalizations.
- **Test-Time Compute**: PRMs are the evaluation engine that powers test-time compute strategies like beam search and Tree-of-Thought.
- **Tree-of-Thought**: ToT needs a way to evaluate intermediate reasoning states. PRMs provide exactly this capability.
- **RLVR**: RLVR uses verifiable final outcomes as rewards (like an ORM with perfect accuracy). PRMs complement RLVR by adding step-level signals to the outcome-level verification.
- **DPO**: DPO can be applied at the step level (StepDPO), preferring correct reasoning steps over incorrect ones, combining the simplicity of DPO with the density of PRM-style supervision.

## Diagrams and Visualizations

*Recommended visual: Process Reward Model vs Outcome Reward Model showing step-level vs final-answer evaluation — see [Let's Verify Step by Step Paper (arXiv:2305.20050)](https://arxiv.org/abs/2305.20050)*

*Recommended visual: PRM scoring each reasoning step in a math problem with step-level correctness labels — see [OpenAI Process Reward Models Blog](https://openai.com/index/improving-mathematical-reasoning-with-process-reward-models/)*

## Further Reading

- Lightman et al., "Let's Verify Step by Step" (OpenAI, 2023) -- the foundational PRM paper, introducing PRM800K and demonstrating that process supervision outperforms outcome supervision for mathematical reasoning.
- Uesato et al., "Solving Math Word Problems with Process- and Outcome-Based Feedback" (DeepMind, 2022) -- systematic comparison of process and outcome supervision, showing complementary strengths.
- Wang et al., "Math-Shepherd: Verify and Reinforce LLMs Step-by-step without Human Annotations" (2024) -- demonstrates automatic PRM training through Monte Carlo estimation, removing the need for expensive human step labels.
- Snell et al., "Scaling LLM Test-Time Compute Optimally Can Be More Effective Than Scaling Model Parameters" (2024) -- shows how PRMs enable effective test-time compute scaling strategies.
