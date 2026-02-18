# Reasoning Models (o1/R1 Paradigm)

**One-Line Summary**: Reasoning models perform extended internal deliberation before answering, trading additional inference-time compute for dramatically improved accuracy on math, code, and science tasks.

**Prerequisites**: Transformer architecture, reinforcement learning from human feedback (RLHF), chain-of-thought prompting, scaling laws

## What Are Reasoning Models?

Imagine two chess players. One plays speed chess -- glancing at the board and moving within seconds. The other studies the board for minutes, mentally simulating dozens of move sequences, evaluating positions, and backtracking from dead ends before committing to a move. Both know the same rules, but the deliberate player wins more often on complex positions. Reasoning models are the deliberate player: they invest significant inference-time compute to think through problems step-by-step before producing a final answer.

Traditional LLMs generate answers in a single forward pass or a short chain-of-thought, essentially "blurting out" the first plausible response. Reasoning models instead produce long internal reasoning chains -- sometimes thousands of tokens -- exploring multiple approaches, catching their own errors, backtracking from failed attempts, and verifying solutions before committing to a final response. This represents a fundamental paradigm shift from scaling training compute (bigger models, more data) to scaling inference compute (more thinking time per query). The bet is that a moderately-sized model that thinks carefully can outperform a much larger model that answers impulsively.

The two landmark systems that defined this paradigm are OpenAI's o1 (September 2024) and DeepSeek's R1 (January 2025). OpenAI's approach keeps the reasoning chain hidden from users; DeepSeek's open-weight R1 exposes the full chain in `<think>` tags. Together, they demonstrate that inference-time scaling can yield performance jumps comparable to orders-of-magnitude increases in model size, fundamentally changing how we think about making LLMs smarter.

## How It Works

### OpenAI o1: Reinforcement Learning for Reasoning

OpenAI's o1 was trained using large-scale reinforcement learning to produce and refine internal chains of thought. The model learns to allocate more computation to harder problems -- spending more tokens "thinking" when the problem demands it and less on easy queries. The reasoning chain is hidden from users for safety and competitive reasons; only the final answer is shown, along with a brief summary of the reasoning steps taken.

Key results for o1:

- **AIME 2024** (competition math): 83.3% vs GPT-4o's 13.4% -- a 6x improvement
- **MATH benchmark**: 94.8% accuracy, up from GPT-4o's ~76%
- **Codeforces**: 93rd percentile rating, competitive with strong human programmers
- **GPQA (PhD-level science)**: 77.3%, approaching human expert level on graduate-level questions

The successor, o3, pushed even further: 87.5% on ARC-AGI (a general reasoning benchmark previously thought to require fundamental architectural breakthroughs), 96.7% on MATH, and 25.2% on FrontierMath (a benchmark of unsolved research-level math problems where all previous models scored near 0%).

### DeepSeek-R1: Open-Weight Reasoning

DeepSeek-R1 revealed the full training pipeline for building reasoning models, using a carefully designed 4-stage process:

**Stage 1 -- R1-Zero (Pure RL)**: Starting from the base DeepSeek-V3 model, they applied Group Relative Policy Optimization (GRPO) with only correctness-based rewards. No supervised fine-tuning, no human demonstrations of reasoning. Remarkably, the model spontaneously developed reasoning behaviors including self-verification ("let me check this"), reflection ("wait, that approach is wrong"), and what the authors called "aha moments" -- instances where the model discovered novel problem-solving strategies during RL training that were never demonstrated in the training data.

**Stage 2 -- Cold-Start SFT**: R1-Zero produced correct answers but with poor formatting, language mixing (switching between Chinese and English mid-chain), and rambling reasoning. To address this, they fine-tuned on a small curated dataset of high-quality chain-of-thought examples, giving the subsequent RL stage a cleaner starting point.

**Stage 3 -- Large-Scale RL**: Applied reasoning-focused RL across math, code, science, and logic tasks. Used both rule-based rewards (correctness of final answer) and learned reward models (reasoning quality, helpfulness, safety). This stage refined the reasoning capabilities into robust, generalizable skills.

**Stage 4 -- Rejection Sampling + Final SFT**: Generated many candidate solutions with the RL-trained model, filtered to keep only the highest-quality outputs (correct answers with clear, well-structured reasoning), and performed a final round of supervised fine-tuning. This consolidation step locks in the reasoning capabilities while ensuring consistent output quality and formatting.

Results: R1 matched o1 on MATH (97.3%) and AIME 2024 (79.8%), demonstrating that fully open-weight models could achieve frontier reasoning performance with a well-designed training pipeline. This was a watershed moment: it showed that the reasoning capability was not a proprietary secret but a reproducible training methodology that any well-resourced team could implement.

### Hidden vs Visible Reasoning

This is a significant design choice with real consequences for the entire ecosystem:

**OpenAI's approach (hidden)**: o1 hides reasoning chains, showing users only a brief summary like "Thought for 32 seconds." The rationale includes protecting proprietary training methods, preventing adversarial prompt injection targeting the reasoning steps, and maintaining a cleaner user experience. The downside is that users cannot verify the reasoning, debug errors in the chain, or learn from the model's problem-solving process.

**DeepSeek's approach (visible)**: R1 exposes the full chain in `<think>...</think>` tags, letting users read every step of the model's deliberation. This enables interpretability, debugging, educational use, and trust-building. Researchers can study how the model reasons, identify failure patterns, and build on the approach. The downside is potential vulnerability to adversarial exploitation and a noisier user experience for non-technical users.

The industry is split on which approach is better, and some models now offer both options -- showing or hiding reasoning on a per-request basis depending on the use case.

### Distillation: Reasoning for Smaller Models

One of R1's most impactful contributions was demonstrating that reasoning capabilities can be distilled into much smaller models. By fine-tuning smaller models on R1's reasoning traces:

- **R1-Distill-Qwen-14B**: 93.9% on MATH, outperforming GPT-4o despite being a far smaller model
- **R1-Distill-Qwen-7B**: Strong reasoning at a fraction of the compute cost, suitable for on-device deployment
- **R1-Distill-Llama-8B**: Demonstrated that reasoning transfers across model architectures, not just within the same model family
- **R1-Distill-Qwen-1.5B**: Even at 1.5 billion parameters, showed meaningful reasoning improvement over the base model, though with clear capability limits on the hardest problems

This proved that reasoning is a learnable behavior pattern -- a skill that can be taught through data -- not just an emergent property of massive scale that requires hundreds of billions of parameters. The implication is profound: frontier reasoning may become accessible on consumer hardware within a few years as distillation techniques improve.

### The Inference-Time Scaling Paradigm

The traditional approach to improving LLMs was training-time scaling: use more data, more parameters, more compute during training. Reasoning models introduce a complementary axis: inference-time scaling. Instead of (or in addition to) making the model bigger, you let it think longer on each query.

This has practical implications for deployment:

- Easy questions can be answered quickly with minimal reasoning, keeping costs low
- Hard questions automatically receive more compute, improving accuracy where it matters most
- The cost-accuracy tradeoff can be tuned per query, per user, or per application
- A single model can serve both simple chatbot queries and complex math problems by adjusting the reasoning budget

This adaptive compute allocation is analogous to how humans spend more mental effort on hard problems and answer easy ones reflexively. It enables a much more efficient use of compute resources compared to fixed-cost-per-query models.

## Why It Matters

1. **Paradigm shift**: Reasoning models invert the scaling paradigm -- instead of only making models bigger at training time, we can make them think longer at inference time, offering a new axis for improvement.
2. **Unlocking hard problems**: Performance on competition math, PhD-level science, and competitive programming jumped from amateur to expert level, opening application domains previously out of reach for AI.
3. **Distillation potential**: Reasoning capabilities can be transferred to small, deployable models, making frontier-level reasoning practical for edge deployment and cost-sensitive applications.
4. **Emergent behaviors**: R1-Zero's spontaneous development of self-verification and reflection without any human demonstrations suggests that reasoning may be a natural consequence of RL optimization pressure on language models.
5. **Open research**: DeepSeek-R1's publication of the full training recipe -- including the GRPO algorithm, the 4-stage pipeline, and the distillation methodology -- enables the broader research community to build on and improve reasoning model techniques.

## Key Technical Details

- o1's reasoning chains can run to 10,000+ tokens internally before producing a response, making it 10-100x more expensive per query than standard models
- GRPO (Group Relative Policy Optimization) avoids the need for a separate critic model by computing advantages relative to the group of sampled responses, reducing training cost
- R1-Zero exhibited language mixing and poor formatting before SFT, showing that pure RL optimizes for correctness at the expense of human readability and presentation
- Inference-time scaling follows a log-linear relationship: doubling the number of thinking tokens yields roughly constant accuracy gains across benchmarks
- Reasoning models show the largest improvements on tasks with verifiable answers (math, code, formal logic) and smaller gains on open-ended or subjective tasks
- Test-time compute can be allocated adaptively: easy problems get short reasoning chains, hard problems get long chains, optimizing the cost-per-query tradeoff
- The "aha moment" phenomenon in R1-Zero training shows the model re-evaluating its approach mid-chain with phrases like "Wait, let me reconsider," analogous to human insight
- Reasoning chains exhibit recognizable problem-solving strategies: decomposition, case analysis, proof by contradiction, and working backwards from the desired result
- Latency is significantly higher than standard models: o1 can take 30-120 seconds per response on hard problems, compared to 1-5 seconds for GPT-4o
- Token-level pricing for reasoning models typically charges for both the hidden reasoning tokens and the visible output tokens, making cost estimation harder

## Common Misconceptions

- **"Reasoning models are just chain-of-thought prompting."** While they produce chains of thought, the chains are generated by models specifically trained via RL to reason effectively. A standard model prompted for CoT produces qualitatively different (and substantially worse) reasoning on hard problems compared to o1 or R1. The RL training teaches the model when to backtrack, verify, and explore alternatives.
- **"More thinking always means better answers."** There are diminishing returns to inference-time compute. On simple factual questions or easy tasks, extended reasoning adds cost without improving accuracy and can even degrade it through overthinking. Reasoning models are most valuable on problems that genuinely require multi-step deliberation.
- **"DeepSeek-R1 replicated o1 exactly."** While R1 achieves comparable benchmark scores, the training procedures likely differ in significant ways. DeepSeek published their approach; OpenAI's remains proprietary. The convergent results suggest similar underlying principles but not identical methods or architectures.
- **"Reasoning models make standard LLMs obsolete."** For the majority of LLM use cases -- summarization, translation, simple Q&A, content generation -- standard models are faster, cheaper, and equally effective. Reasoning models add value specifically on hard analytical tasks where deliberation improves accuracy.

## Connections to Other Concepts

- **Chain-of-Thought Prompting**: Reasoning models formalize and optimize the CoT approach through RL training, producing higher-quality and more reliable reasoning chains than prompting alone can achieve.
- **Reinforcement Learning from Human Feedback (RLHF)**: Reasoning models extend RLHF by using correctness-based rewards (math proofs, code test suites) alongside or instead of human preference signals.
- **Scaling Laws**: Reasoning models introduce inference-time scaling as a complement to training-time scaling, adding a new dimension to the compute-performance tradeoff.
- **Self-Reflection**: The self-verification behavior that emerges naturally in reasoning models is closely related to self-reflection, but occurs within a single generation rather than across separate trials.
- **Distillation and Model Compression**: The R1 distillation results demonstrate that reasoning can be transferred from large teacher models to small student models, extending classical knowledge distillation techniques to emergent reasoning behaviors.

## Further Reading

- DeepSeek-AI, "DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning," arXiv:2501.12948, 2025
- OpenAI, "Learning to Reason with LLMs," OpenAI Blog, September 2024
- Snell et al., "Scaling LLM Test-Time Compute Optimally Can Be More Effective Than Scaling Model Parameters," arXiv:2408.03314, 2024
- Zelikman et al., "STaR: Bootstrapping Reasoning With Reasoning," NeurIPS 2022
- Lightman et al., "Let's Verify Step by Step," arXiv:2305.20050, 2023
