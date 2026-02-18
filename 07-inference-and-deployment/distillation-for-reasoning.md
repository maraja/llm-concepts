# Distillation for Reasoning

**One-Line Summary**: Distillation for reasoning transfers chain-of-thought reasoning capabilities from large teacher models to smaller student models by training on the teacher's detailed reasoning traces -- enabling results like DeepSeek-R1-Distill-Qwen-7B scoring 55.5% on AIME 2024 and R1-Distill-Qwen-14B achieving 93.9% on MATH, with a critical finding that distillation outperforms direct RL training at small model scales.

**Prerequisites**: Knowledge distillation fundamentals (teacher-student framework, soft labels), chain-of-thought prompting and reasoning traces, supervised fine-tuning, reinforcement learning for language models (RLHF, GRPO), and the distinction between model capabilities and model scale.

## What Is Distillation for Reasoning?

Standard knowledge distillation transfers a large model's knowledge through soft probability distributions over the vocabulary -- the teacher's output logits guide the student's learning. But reasoning is different from knowledge. A model that can solve a complex math problem does not just need the right answer; it needs the right *thinking process*. The answer "42" is meaningless without the chain of logical steps that produced it.

Distillation for reasoning adapts the teacher-student framework to transfer *process*, not just *output*. Instead of training the student to match the teacher's token-level probability distributions, you generate the teacher's complete chain-of-thought traces -- the full step-by-step reasoning from problem to solution -- and train the student to reproduce these reasoning patterns through standard supervised fine-tuning.

The analogy is the difference between copying a textbook's answer key versus studying worked solutions. A student who memorizes answers learns nothing about problem-solving. A student who studies hundreds of worked solutions, seeing how an expert approaches different types of problems, develops genuine reasoning skills that transfer to novel problems. Reasoning distillation gives small models hundreds of thousands of "worked solutions" from a much larger, more capable teacher.

## How It Works

### The Reasoning Trace Pipeline

The distillation process follows a generate-then-train pipeline:

1. **Curate prompts**: Collect a diverse set of problems requiring reasoning (math, coding, logic, science). Quality and diversity of prompts are as important as volume.
2. **Generate traces**: Run the teacher model on each prompt, generating complete chain-of-thought reasoning traces. These traces include intermediate steps, self-corrections, explorations of dead ends, and verification.
3. **Filter traces**: Verify that the traces reach correct final answers using ground-truth labels, automated checkers, or reward models. Incorrect traces are discarded.
4. **Fine-tune the student**: Train the smaller model on the filtered traces using standard supervised fine-tuning (cross-entropy loss on the full reasoning sequence).

The student learns not just the final answers but the reasoning patterns, heuristics, and problem-solving strategies embedded in the teacher's traces.

### DeepSeek-R1 Distillation at Scale

DeepSeek's R1 distillation is the most comprehensive demonstration of reasoning distillation to date. The pipeline produced:

1. **Teacher**: DeepSeek-R1 (a 671B MoE model trained with GRPO on reasoning tasks).
2. **Trace generation**: ~800,000 high-quality reasoning traces across mathematics, coding, science, and logical reasoning.
3. **Students**: Six models distilled from the traces:
   - Qwen-1.5B, Qwen-7B, Qwen-14B, Qwen-32B (based on Qwen 2.5)
   - Llama-8B, Llama-70B (based on Llama 3)
4. **Training**: Standard SFT on the reasoning traces, with no RL stage for the distilled models.

The results were striking. R1-Distill-Qwen-7B achieved 55.5% on AIME 2024 (competition mathematics) and 92.3% on MATH. R1-Distill-Qwen-14B reached 93.9% on MATH. These scores significantly exceeded what was achievable by training models of the same size with direct RL.

### The Distillation vs. Direct RL Finding

Perhaps the most important finding from DeepSeek-R1: **at small model scales, distillation dramatically outperforms direct reinforcement learning**. When DeepSeek tried applying GRPO directly to Qwen-7B (without distillation), the resulting model scored substantially lower than the distilled version trained on R1's traces.

The explanation is intuitive: small models may not have sufficient capacity to *discover* effective reasoning strategies through RL exploration, but they have sufficient capacity to *imitate* strategies demonstrated in the teacher's traces. RL requires the model to find good reasoning paths through trial and error; distillation provides those paths directly.

### Orca: Teaching Reasoning Strategies

Microsoft's Orca series pioneered a complementary approach -- distilling not just reasoning traces but explicit reasoning strategies:

**Orca 1**: Collected "explanation traces" from GPT-4 -- responses where the teacher was prompted to explain its reasoning step-by-step, show its work, and provide pedagogical explanations. A 13B student trained on these traces significantly outperformed GPT-3.5 on reasoning benchmarks.

**Orca 2**: Went further by teaching the student *when to use which reasoning strategy*. The teacher was prompted with different strategies (step-by-step for math, analogical for common sense, elimination for multiple choice), and the student learned to select the appropriate strategy based on the problem type. This meta-reasoning capability meant Orca 2 sometimes outperformed models 5-10x its size.

## Why It Matters

1. **Democratizes reasoning capability**: Distilled 7B and 14B models can run on consumer GPUs while achieving reasoning performance that previously required models 10-100x larger, making advanced reasoning accessible outside large labs.
2. **Distillation outperforms direct RL at small scales**: This finding reshapes how small reasoning models should be built. Rather than investing in RL infrastructure for small models, labs can invest in generating high-quality reasoning traces from large teachers.
3. **Transfers process, not just knowledge**: Unlike standard distillation that transfers soft labels, reasoning distillation transfers problem-solving strategies, heuristics, and meta-cognitive patterns like self-verification and backtracking.
4. **Efficient training**: SFT on reasoning traces is dramatically simpler and more stable than RL training. No reward models, no critic networks, no PPO hyperparameter tuning -- just standard supervised learning.
5. **Compounding capabilities**: A strong distilled model can serve as the teacher for the next generation of even-smaller students, creating a cascade of reasoning capability transfer.

## Key Technical Details

- **DeepSeek-R1 distilled models performance**: R1-Distill-Qwen-1.5B: 28.9% AIME, 83.9% MATH. R1-Distill-Qwen-7B: 55.5% AIME, 92.3% MATH. R1-Distill-Qwen-14B: 69.7% AIME, 93.9% MATH. R1-Distill-Llama-70B: 70.0% AIME, 94.5% MATH.
- **Trace volume**: ~800K reasoning traces from DeepSeek-R1, filtered for correctness. Quality and diversity of traces matters more than raw volume.
- **Trace length**: Reasoning traces average 500-2000 tokens, with complex math problems producing traces of 3000+ tokens. Students must handle these long reasoning sequences.
- **No RL needed for students**: The distilled models achieve their performance purely through SFT on traces. Applying RL on top of distillation provides only marginal additional gains.
- **Orca 1 results**: The 13B Orca model matched GPT-3.5 on multiple reasoning benchmarks, despite being ~10x smaller in parameters.
- **Orca 2 key insight**: Teaching a small model *when* to use different reasoning strategies (not just *how*) was critical for closing the gap with much larger models.

## Common Misconceptions

- **"Distillation just teaches the student to copy the teacher's answers."** Reasoning distillation trains on the full chain-of-thought, not just final answers. The student learns problem-solving processes that generalize to novel problems not seen during training.
- **"Smaller models cannot really reason; they just memorize patterns."** The distilled models' strong performance on held-out benchmarks (like AIME problems from 2024, after their training data cutoff) demonstrates genuine generalization, not memorization.
- **"Direct RL is always better because the model discovers its own strategies."** At small scales, the model's capacity to explore and discover is limited. Distillation provides a strong starting point that RL alone cannot reach. The finding from DeepSeek-R1 is clear: distillation > direct RL for small models.
- **"You need the teacher's logits for effective distillation."** Reasoning distillation works with only the teacher's text output (the reasoning traces). No access to the teacher's internal logits, weights, or architecture is required, making it applicable even with closed-source teachers.

## Connections to Other Concepts

- **Knowledge distillation**: Reasoning distillation is a specialized form of knowledge distillation, focused on transferring reasoning processes rather than output distributions. Standard distillation uses soft logits; reasoning distillation uses explicit chain-of-thought traces.
- **Chain-of-thought training**: The traces used for distillation are chain-of-thought sequences. Understanding CoT prompting and training is essential context for reasoning distillation.
- **GRPO and RL for reasoning**: DeepSeek-R1 (the teacher) was trained with GRPO. The finding that distillation outperforms direct RL at small scales informs the choice between these training paradigms.
- **Supervised fine-tuning**: The actual training step of reasoning distillation is standard SFT. The novelty is in the data (reasoning traces from a teacher) rather than the training algorithm.
- **Self-play and self-improvement**: Reasoning distillation is teacher-dependent, while self-improvement methods bootstrap from the model's own outputs. These approaches are complementary -- a distilled model can be further improved through self-play.

## Further Reading

1. **"DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning" (DeepSeek-AI, 2025, arXiv:2501.12948)** -- Presents the comprehensive reasoning distillation pipeline, the 800K trace dataset, six distilled models, and the critical finding that distillation outperforms direct RL at small scales.
2. **"Orca: Progressive Learning from Complex Explanation Traces of GPT-4" (Mukherjee et al., 2023, arXiv:2306.02707)** -- Pioneers the concept of explanation trace distillation, showing that rich teacher explanations produce stronger students than standard distillation.
3. **"Orca 2: Teaching Small Language Models How to Reason" (Mitra et al., 2023, arXiv:2311.11045)** -- Extends Orca with strategy selection, teaching students not just how to reason but when to apply different reasoning approaches.
