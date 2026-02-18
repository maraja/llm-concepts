# Self-Play and Self-Improvement

**One-Line Summary**: Self-play and self-improvement methods enable language models to bootstrap stronger capabilities from their own outputs -- generating reasoning traces, filtering for correctness, and training on the successes -- achieving dramatic gains like GPT-J 6B jumping from 36.6% to 72.5% on CommonsenseQA without any human-written rationales.

**Prerequisites**: Supervised fine-tuning and cross-entropy loss, chain-of-thought prompting and reasoning traces, reward models or verifiers for output evaluation, reinforcement learning basics (policy improvement), and the concept of data augmentation.

## What Is Self-Play and Self-Improvement?

Training language models typically requires human-generated data -- instruction-response pairs, reasoning demonstrations, preference labels. But human data is expensive, slow to collect, and fundamentally limited by what humans choose to write down. Self-play and self-improvement methods flip this: the model generates its own training data, filters it for quality, and trains on its own best outputs.

The idea has deep roots in game-playing AI. AlphaGo Zero learned superhuman Go by playing against itself millions of times, never seeing a human game. Language model self-improvement adapts this principle: the model generates candidate solutions to problems, uses some verification mechanism (a math checker, a unit test, a reward model, or even the model itself) to identify which solutions are correct or high-quality, and then trains on those verified outputs. Each iteration produces a better model that generates better candidates in the next round.

This creates a virtuous cycle -- but also a fragile one. Unlike board games with perfect verifiers (the rules determine who wins), language tasks often lack reliable verification. A model that trains on its own outputs risks amplifying its own biases and errors, leading to "model collapse" where diversity decreases and quality plateaus or degrades. The methods described here represent different strategies for managing this tension between self-improvement and self-corruption.

## How It Works

### STaR (Self-Taught Reasoner)

STaR (Zelikman et al., 2022) is the foundational method for bootstrapping reasoning through self-improvement:

1. **Generate**: For each question in the training set, the model generates a chain-of-thought rationale followed by an answer.
2. **Filter**: Keep only the rationales that led to correct final answers (verified against ground truth).
3. **Rationalize**: For questions the model got wrong, provide the correct answer as a hint and have the model generate a rationale that reaches it (rationalization).
4. **Train**: Fine-tune the model on the combined set of correct rationales and rationalized examples.
5. **Iterate**: Repeat from step 1 with the improved model.

The rationalization step is crucial -- it prevents the model from only improving on problems it can already solve. By conditioning on the correct answer, the model learns reasoning patterns that were just beyond its reach. Results: GPT-J 6B improved from 36.6% to 72.5% on CommonsenseQA and from 10.9% to 36.3% on GSM8K.

### Quiet-STaR (Internalized Reasoning)

Quiet-STaR (Zelikman et al., 2024) extends STaR by having the model generate internal "thoughts" before every token in a sequence, not just at the beginning of a response:

1. At each token position, the model generates a short internal thought (a sequence of "thought tokens").
2. A mixing head learns to combine the post-thought hidden state with the pre-thought hidden state.
3. The model is trained to minimize next-token prediction loss, with the internal thoughts treated as latent variables.
4. A REINFORCE-based gradient estimator is used since thought generation is discrete and non-differentiable.

This converts chain-of-thought from an explicit output format into an implicit internal computation. Results: Mistral 7B improved by +10.9% on GSM8K and +3.8% on CommonsenseQA, with the internal thoughts showing meaningful reasoning patterns despite never being explicitly trained to reason.

### SPIN (Self-Play Fine-Tuning)

SPIN (Chen et al., 2024) frames self-improvement as a two-player game:

1. **Generator**: The current model generates responses to prompts.
2. **Discriminator**: The same model (or a training objective) learns to distinguish between model-generated responses and human-written ground truth responses.
3. **Training**: Update the model to make its generated responses indistinguishable from human data, using a DPO-like objective where human responses are preferred over model responses.
4. **Convergence**: The game has a fixed point -- when the model's distribution exactly matches the human data distribution, the discriminator can no longer distinguish them, and training converges.

SPIN provides theoretical convergence guarantees: training provably stops improving when the model matches the target distribution, preventing over-optimization.

### ReST and ReST^EM

ReST (Reinforcement from Self-Training) and its variant ReST^EM use an expectation-maximization-inspired approach:

1. **E-step (Generate)**: Sample many outputs from the current model for each prompt.
2. **M-step (Improve)**: Filter outputs using a reward model or verifier, then fine-tune on the top-scoring outputs.
3. **Iterate**: Repeat with the improved model.

This is closely related to rejection sampling fine-tuning but formalized as an EM algorithm, with theoretical connections to variational inference.

## Why It Matters

1. **Data efficiency**: Self-improvement can extract substantially more capability from existing training data by generating and leveraging model-produced reasoning traces, effectively augmenting the dataset with synthetic demonstrations.
2. **Beyond human demonstrations**: Models can potentially discover reasoning strategies that humans would not naturally demonstrate, especially on technical or mathematical problems where the model explores a broader space of solution approaches.
3. **Scalability**: Once a verifier exists (math checker, code executor, reward model), self-improvement can run indefinitely with compute as the only bottleneck, unlike human data collection which has inherent throughput limits.
4. **Emergent capabilities**: Self-play can elicit capabilities that were latent in the model but not expressed. STaR showed that models "know" more than they demonstrate without the right reasoning scaffolding.
5. **Complementary to RL**: Self-improvement via data generation and filtering is often more stable than direct RL optimization, and the two approaches can be combined (as in DeepSeek-R1's pipeline).

## Key Technical Details

- **STaR results**: GPT-J 6B from 36.6% to 72.5% on CommonsenseQA (nearly 2x improvement) and 10.9% to 36.3% on GSM8K (3.3x improvement) over multiple iterations.
- **Quiet-STaR**: +10.9% on GSM8K and +3.8% on CommonsenseQA for Mistral 7B, with internal thoughts of 1-5 tokens providing meaningful signal.
- **SPIN**: After 3 iterations, converges and provides no further improvement -- matching theoretical predictions. Gains are concentrated in the first 1-2 iterations.
- **Collapse risk**: Without external verification, self-improvement degrades after 3-5 iterations as the model's output distribution narrows and errors accumulate.
- **Verification requirement**: Domains with reliable verifiers (math, code, logic puzzles) show the strongest self-improvement gains. Open-ended generation tasks remain challenging.
- **Diminishing returns**: Each iteration provides less improvement than the previous one. Most gains come in the first 2-3 iterations, with subsequent iterations showing marginal returns.

## Common Misconceptions

- **"Self-improvement means the model can improve infinitely."** All methods show diminishing returns, and without external verification or new data, the model eventually plateaus or degrades. The improvement is bounded by the model's latent capabilities and the quality of verification.
- **"Training on model outputs always leads to model collapse."** Collapse occurs when training on unfiltered model outputs. With proper filtering (verification, reward models), self-improvement avoids collapse by selecting only high-quality outputs. The filtering mechanism is what makes the difference.
- **"Self-play requires two separate models."** In SPIN and STaR, the same model plays both roles (generator and evaluator/trainee), just used in different modes. No separate opponent model is needed.
- **"This is just data augmentation."** Self-improvement fundamentally changes the model's behavior through iterative refinement. Each iteration builds on the previous, creating a compounding effect that simple one-shot data augmentation does not achieve.

## Connections to Other Concepts

- **Chain-of-thought training**: STaR and Quiet-STaR are methods for training chain-of-thought capability, bridging the gap between prompting-based CoT and natively trained reasoning.
- **Rejection sampling**: ReST and iterative rejection sampling fine-tuning are closely related -- both generate many outputs, filter for quality, and train on the best. Rejection sampling is a building block within self-improvement pipelines.
- **GRPO and RLVR**: DeepSeek-R1's approach can be viewed as self-improvement through RL -- the model generates outputs, receives reward signals, and updates its policy, with emergent reasoning as the result.
- **Synthetic data**: Self-improvement is a structured form of synthetic data generation, where the model creates its own training data through a principled generate-filter-train loop.
- **DPO**: SPIN uses a DPO-like objective for self-play, connecting self-improvement to preference optimization.

## Further Reading

1. **"STaR: Bootstrapping Reasoning With Reasoning" (Zelikman et al., 2022, arXiv:2203.14465)** -- Introduces the foundational generate-filter-rationalize-train loop for self-taught reasoning, with impressive results on GPT-J.
2. **"Self-Play Fine-Tuning Converts Weak Language Models to Strong Language Models" (Chen et al., 2024, arXiv:2401.01335)** -- Formalizes self-play for language models with convergence guarantees, providing theoretical grounding for the approach.
3. **"ReST meets ReAct: Self-Improvement for Multi-Step Reasoning LLM Agent" (Aksitov et al., 2023, arXiv:2308.08998)** -- Applies EM-based self-improvement to multi-step reasoning tasks, demonstrating the generality of the generate-filter-train paradigm.
4. **"Quiet-STaR: Language Models Can Teach Themselves to Think Before Speaking" (Zelikman et al., 2024, arXiv:2403.09629)** -- Extends STaR to internalized reasoning, a conceptually novel approach to embedding thought processes within the model's forward pass.
