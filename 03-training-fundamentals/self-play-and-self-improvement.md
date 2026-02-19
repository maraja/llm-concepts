# Self-Play and Self-Improvement

**One-Line Summary**: Self-play and self-improvement methods enable language models to bootstrap stronger capabilities from their own outputs -- generating reasoning traces, filtering for correctness, and training on the successes -- achieving dramatic gains like GPT-J 6B jumping from 36.6% to 72.5% on CommonsenseQA without any human-written rationales.

**Prerequisites**: Supervised fine-tuning, chain-of-thought prompting and reasoning traces, reward models or verifiers for output evaluation, reinforcement learning basics (policy improvement), and the concept of synthetic data generation.

## What Is Self-Play and Self-Improvement?

Training language models typically requires human-generated data. But human data is expensive, slow to collect, and limited by what humans choose to write down.

*See the STaR self-improvement loop diagram in: [Zelikman et al., "STaR: Bootstrapping Reasoning With Reasoning" (arXiv:2203.14465)](https://arxiv.org/abs/2203.14465), Figure 1, which illustrates the generate-filter-rationalize-train cycle where the model produces chain-of-thought rationales, filters for correct answers, and retrains on its own verified outputs.*


Self-play and self-improvement methods flip this: the model generates its own training data, filters it for quality, and trains on its own best outputs.

The idea has deep roots in game-playing AI. AlphaGo Zero learned superhuman Go by playing against itself millions of times, never seeing a human game. Language model self-improvement adapts this: the model generates candidate solutions, uses a verification mechanism (math checker, unit test, reward model) to identify correct ones, and trains on those verified outputs.

Each iteration produces a better model that generates better candidates in the next round. But this cycle is fragile. Unlike board games with perfect verifiers, language tasks often lack reliable verification. A model training on its own unfiltered outputs risks "model collapse" -- narrowing output diversity and amplifying errors. The methods below represent different strategies for managing this tension.

## How It Works


*See the SPIN self-play framework in: [Chen et al., "Self-Play Fine-Tuning Converts Weak Language Models to Strong Language Models" (arXiv:2401.01335)](https://arxiv.org/abs/2401.01335), Figure 1, which shows the two-player game between the generator and discriminator, with convergence occurring when model outputs become indistinguishable from human data.*

### STaR (Self-Taught Reasoner)

STaR (Zelikman et al., 2022) is the foundational method:

1. **Generate**: For each question, the model generates a chain-of-thought rationale followed by an answer.
2. **Filter**: Keep only rationales that led to correct final answers.
3. **Rationalize**: For questions the model got wrong, provide the correct answer as a hint and have the model generate a rationale reaching it.
4. **Train**: Fine-tune on the combined correct and rationalized examples.
5. **Iterate**: Repeat with the improved model.

The rationalization step is crucial. Without it, the model only improves on problems it can already solve. By conditioning on the correct answer, STaR teaches reasoning patterns for problems *just beyond* the model's current reach.

Results: GPT-J 6B improved from 36.6% to 72.5% on CommonsenseQA and from 10.9% to 36.3% on GSM8K, all without human-written chain-of-thought demonstrations.

### Quiet-STaR (Internalized Reasoning)

Quiet-STaR (Zelikman et al., 2024) generates internal "thoughts" before every token, not just at response start:

1. At each token position, generate a short internal thought (special "thought tokens" invisible to the user).
2. A mixing head combines post-thought and pre-thought hidden states.
3. Train to minimize next-token prediction loss, with thoughts as latent variables.
4. Use REINFORCE for gradients through the discrete thought generation.

This converts chain-of-thought from visible output to implicit internal computation.

Results on Mistral 7B: +10.9% GSM8K, +3.8% CommonsenseQA, with 1-5 token thoughts providing meaningful signal.

### SPIN (Self-Play Fine-Tuning)

SPIN (Chen et al., 2024) frames improvement as a two-player game:

1. **Generator**: The model generates responses to prompts.
2. **Discriminator**: A DPO-like objective learns to distinguish model outputs from human ground truth.
3. **Update**: The model is trained to make its outputs indistinguishable from human data.
4. **Convergence**: When the model's distribution matches human data, the discriminator cannot distinguish them and training stops.

SPIN provides theoretical convergence guarantees: it provably converges when the model matches the target distribution.

### ReST and ReST^EM

ReST uses an EM-inspired framework:

*See also the Quiet-STaR internalized reasoning architecture at: [Zelikman et al., "Quiet-STaR" (arXiv:2403.09629)](https://arxiv.org/abs/2403.09629), Figure 1, which shows how thought tokens are generated at each position and a mixing head combines post-thought and pre-thought hidden states.*


1. **E-step (Generate)**: Sample many outputs from the current model.
2. **M-step (Improve)**: Filter with a reward model, fine-tune on top-scoring outputs.
3. **Iterate**: Generate again with the improved model.

The EM framing connects to variational inference: the E-step samples from the model's posterior over good solutions, and the M-step maximizes their likelihood.

## Why It Matters

1. **Data efficiency**: Self-improvement extracts substantially more capability from existing data by generating synthetic reasoning traces.
2. **Beyond human demonstrations**: Models can discover reasoning strategies humans would not naturally write down.
3. **Scalability**: Given a verifier, self-improvement runs indefinitely with compute as the only bottleneck.
4. **Emergent capabilities**: Self-play elicits latent capabilities the model does not express in standard generation.
5. **Complementary to RL**: Generate-filter-train is often more stable than direct RL, and the two combine effectively.

## Key Technical Details

- **STaR**: GPT-J 6B 36.6% → 72.5% CommonsenseQA, 10.9% → 36.3% GSM8K.
- **Quiet-STaR**: +10.9% GSM8K, +3.8% CommonsenseQA on Mistral 7B.
- **SPIN**: Converges after 3 iterations, matching theoretical fixed-point predictions.
- **Collapse risk**: Without verification, degradation after 3-5 iterations as errors accumulate.
- **Verification requirement**: Math, code, logic show strongest gains. Open-ended generation remains harder.
- **Diminishing returns**: Most gains in first 2-3 iterations; subsequent iterations show rapidly decreasing marginal returns.
- **Rationalization quality**: STaR's backward-generated rationales are lower quality than naturally correct ones but still provide useful signal.
- **Compute scaling**: Each iteration costs approximately one SFT run plus one generation pass over the training set.
- **Filtering threshold**: Too strict yields too little data; too lenient introduces noise. Adaptive thresholds help.

## Common Misconceptions

- **"Self-improvement means infinite improvement."** All methods show diminishing returns. Without verification or new data, the model plateaus or degrades. Improvement is bounded by latent capabilities and verification quality.
- **"Training on model outputs always causes collapse."** Collapse occurs with *unfiltered* outputs. Proper filtering through verification avoids it. The filtering mechanism is the essential difference.
- **"Self-play requires two separate models."** In SPIN, STaR, and ReST, the same model plays both roles (generator and trainee) in different modes.
- **"This is just data augmentation."** Self-improvement is *iterative* -- each round builds on the previous, creating compounding effects that one-shot augmentation does not achieve.

## Connections to Other Concepts

- **Chain-of-thought training**: STaR and Quiet-STaR are methods for training CoT capability, bridging prompting-based CoT and natively trained reasoning.
- **Rejection sampling**: ReST and iterative RS fine-tuning are closely related -- both generate, filter, and train on the best outputs.
- **GRPO and RLVR**: DeepSeek-R1 can be viewed as self-improvement through RL, with emergent reasoning as the result.
- **Synthetic data**: Self-improvement is structured, iterative synthetic data generation.
- **DPO**: SPIN uses a DPO-like objective, connecting self-improvement to preference optimization.

## Further Reading

1. **"STaR: Bootstrapping Reasoning With Reasoning" (Zelikman et al., 2022, arXiv:2203.14465)** -- The foundational generate-filter-rationalize-train loop for self-taught reasoning.
2. **"Self-Play Fine-Tuning Converts Weak Language Models to Strong Language Models" (Chen et al., 2024, arXiv:2401.01335)** -- Formalizes self-play with convergence guarantees.
3. **"ReST meets ReAct: Self-Improvement for Multi-Step Reasoning LLM Agent" (Aksitov et al., 2023, arXiv:2308.08998)** -- EM-based self-improvement for multi-step reasoning.
4. **"Quiet-STaR: Language Models Can Teach Themselves to Think Before Speaking" (Zelikman et al., 2024, arXiv:2403.09629)** -- Internalized reasoning with thought tokens.
