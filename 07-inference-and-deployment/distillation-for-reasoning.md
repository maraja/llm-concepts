# Distillation for Reasoning

**One-Line Summary**: Distillation for reasoning transfers chain-of-thought reasoning capabilities from large teacher models to smaller student models by training on the teacher's detailed reasoning traces -- enabling results like DeepSeek-R1-Distill-Qwen-7B scoring 55.5% on AIME 2024 and R1-Distill-Qwen-14B achieving 93.9% on MATH, with the critical finding that distillation outperforms direct RL training at small model scales.

**Prerequisites**: Knowledge distillation fundamentals (teacher-student framework, soft label training), chain-of-thought prompting and reasoning traces, supervised fine-tuning and cross-entropy loss, reinforcement learning for language models (RLHF, GRPO), and the distinction between model capabilities and model scale.

## What Is Distillation for Reasoning?

Standard knowledge distillation transfers a large model's knowledge through soft probability distributions over the vocabulary -- the teacher's output logits guide the student's learning, providing richer training signal than hard one-hot labels alone. But reasoning is fundamentally different from factual knowledge. A model that can solve a complex math problem does not just need the right answer; it needs the right *thinking process*. The answer "42" is meaningless without the chain of logical steps that produced it.

![Knowledge distillation teacher-student framework showing how soft labels transfer knowledge from a large model to a small model](https://upload.wikimedia.org/wikipedia/commons/thumb/4/48/Knowledge_Distillation.svg/800px-Knowledge_Distillation.svg.png)
*Source: [Wikimedia Commons - Knowledge Distillation](https://commons.wikimedia.org/wiki/File:Knowledge_Distillation.svg)*


Distillation for reasoning adapts the teacher-student framework to transfer *process*, not just *output*. Instead of training the student to match the teacher's token-level probability distributions, you generate the teacher's complete chain-of-thought traces -- the full step-by-step reasoning from problem statement to final solution -- and train the student to reproduce these reasoning patterns through standard supervised fine-tuning on the trace text.

The analogy is the difference between copying a textbook's answer key versus studying worked solutions. A student who memorizes that the answer to problem 7 is "42" learns nothing transferable. A student who studies hundreds of worked solutions -- seeing how an expert identifies the problem type, selects a strategy, executes intermediate steps, checks the work, and arrives at the conclusion -- develops genuine problem-solving skills that transfer to novel problems never seen before. Reasoning distillation gives small models hundreds of thousands of "worked solutions" from a much larger, more capable teacher model.

## How It Works


*See diagram of the DeepSeek-R1 distillation pipeline and reasoning trace generation at: [DeepSeek-R1 Paper (arXiv:2501.12948)](https://arxiv.org/abs/2501.12948)*

### The Reasoning Trace Pipeline

The distillation process follows a generate-filter-train pipeline:

1. **Curate prompts**: Collect a diverse set of problems requiring reasoning across target domains (mathematics, coding, logical reasoning, science). Quality and diversity of prompts are as important as raw volume -- the prompts should cover different difficulty levels, problem types, and reasoning strategies.
2. **Generate traces**: Run the teacher model on each prompt, generating complete chain-of-thought reasoning traces. These traces typically include problem decomposition, intermediate calculations, exploration of approaches (including dead ends), self-correction, verification steps, and the final answer.
3. **Filter traces**: Verify that the traces reach correct final answers using ground-truth labels, automated checkers (math solvers, code execution), or reward models. Incorrect traces are discarded. Optionally, apply additional quality filtering to remove traces that are correct but poorly structured or unnecessarily verbose.
4. **Fine-tune the student**: Train the smaller model on the filtered traces using standard supervised fine-tuning (next-token prediction with cross-entropy loss on the full reasoning sequence, including all intermediate steps).

The student learns not just the final answers but the reasoning patterns, problem decomposition strategies, self-checking heuristics, and meta-cognitive behaviors embedded in the teacher's reasoning traces.

### DeepSeek-R1 Distillation at Scale

DeepSeek's R1 distillation campaign is the most comprehensive demonstration of reasoning distillation to date, producing six distilled models across two base model families:

1. **Teacher**: DeepSeek-R1, a 671B parameter MoE model trained with GRPO on reasoning tasks using rule-based rewards. The teacher itself exhibited emergent reasoning behaviors including chain-of-thought, self-verification, and backtracking.
2. **Trace generation**: Approximately 800,000 high-quality reasoning traces generated by R1 across mathematics, coding, science, and logical reasoning domains. Each trace was verified for answer correctness before inclusion.
3. **Student models**: Six models distilled from these traces:
   - **Qwen-based**: R1-Distill-Qwen-1.5B, R1-Distill-Qwen-7B, R1-Distill-Qwen-14B, R1-Distill-Qwen-32B (based on Qwen 2.5 base models)
   - **Llama-based**: R1-Distill-Llama-8B, R1-Distill-Llama-70B (based on Llama 3 base models)
4. **Training**: Standard SFT on the reasoning traces with no reinforcement learning stage for the distilled models. The training is conceptually simple -- just next-token prediction on the teacher's reasoning text.

The results were striking and established new benchmarks for small reasoning models. R1-Distill-Qwen-7B achieved 55.5% on AIME 2024 (American Invitational Mathematics Examination, a competition-level math test) and 92.3% on MATH. R1-Distill-Qwen-14B reached 69.7% on AIME and 93.9% on MATH. These scores substantially exceeded what was achievable by training models of the same size with direct reinforcement learning.

### The Critical Finding: Distillation vs. Direct RL

Perhaps the most important finding from the DeepSeek-R1 paper is that **at small model scales, distillation dramatically outperforms direct reinforcement learning**. When DeepSeek applied GRPO directly to Qwen-7B and Qwen-14B base models (without distillation, training them to reason through RL alone), the resulting models scored substantially lower than the distilled versions trained on R1's reasoning traces.

The explanation is intuitive and has significant practical implications: small models may not have sufficient capacity to *discover* effective reasoning strategies through RL exploration alone -- the search space is too vast and the model's exploration is too constrained by its limited capacity. But these same small models have sufficient capacity to *imitate* well-demonstrated strategies from a teacher's traces. RL requires the model to find good reasoning paths through trial and error in an exponentially large space; distillation provides those paths directly as supervised examples.

This finding suggests a practical recipe: train reasoning capability in the largest feasible model using RL (where the model has capacity to explore and discover), then distill the resulting reasoning traces down to smaller models for efficient deployment.

### Orca: Teaching Reasoning Strategies

Microsoft's Orca series pioneered a complementary approach to reasoning distillation, focusing on *explanation quality* and *strategy selection*:

*See diagram of Orca's progressive learning from explanation traces at: [Orca Paper (arXiv:2306.02707)](https://arxiv.org/abs/2306.02707)*


**Orca 1** (Mukherjee et al., 2023): Collected "explanation traces" from GPT-4 where the teacher was specifically prompted to explain its reasoning step-by-step, show all intermediate work, and provide pedagogically clear explanations. Rather than just getting GPT-4's answer with its natural reasoning, the prompts elicited maximally detailed explanations. A 13B student trained on ~5M of these traces significantly outperformed models trained on standard GPT-3.5 outputs and matched GPT-3.5 itself on several reasoning benchmarks.

**Orca 2** (Mitra et al., 2023): Advanced beyond teaching *how* to reason to teaching *when to use which reasoning strategy*. The teacher generated demonstrations using different strategies for different problem types -- step-by-step decomposition for multi-step math, analogical reasoning for common sense questions, process of elimination for multiple choice, direct recall for factual queries. The student learned not just the strategies but a meta-reasoning capability: assessing the problem type and selecting the appropriate strategy. This meant Orca 2 sometimes outperformed models 5-10x its size by using the right tool for each job, rather than applying a single reasoning approach universally.

## Why It Matters

1. **Democratizes advanced reasoning**: Distilled 7B and 14B models can run on consumer GPUs (a single RTX 4090) while achieving reasoning performance that previously required models with 10-100x more parameters running on expensive multi-GPU clusters.
2. **Distillation beats direct RL at small scales**: This finding fundamentally reshapes how the field builds small reasoning models. Rather than investing in complex RL infrastructure for small models, labs can invest in generating high-quality reasoning traces from large teachers -- a simpler and more reliable process.
3. **Transfers reasoning process, not just answers**: Unlike standard distillation that transfers output distributions, reasoning distillation transfers problem-solving strategies, self-verification habits, and meta-cognitive patterns like backtracking and strategy selection.
4. **Simple and stable training**: The student training is just standard SFT -- dramatically simpler and more stable than RL training. No reward models, no critic networks, no PPO hyperparameter sensitivity.
5. **Cascading capability transfer**: A strong distilled model can itself serve as the teacher for even smaller students, creating a cascade of capability transfer across model scales.

## Key Technical Details

- **DeepSeek-R1 distilled model benchmarks**: R1-Distill-Qwen-1.5B: 28.9% AIME, 83.9% MATH. R1-Distill-Qwen-7B: 55.5% AIME, 92.3% MATH. R1-Distill-Qwen-14B: 69.7% AIME, 93.9% MATH. R1-Distill-Qwen-32B: 72.6% AIME, 94.3% MATH. R1-Distill-Llama-70B: 70.0% AIME, 94.5% MATH.
- **Trace volume and diversity**: ~800K reasoning traces from DeepSeek-R1, filtered for correctness. Quality and diversity of traces matter more than raw volume -- traces should span difficulty levels and problem types.
- **Trace length characteristics**: Reasoning traces average 500-2000 tokens, with complex competition-level math problems producing traces of 3000+ tokens including exploration and self-correction. Students must handle these long reasoning sequences without truncation.
- **No RL needed for students**: The distilled models achieve their full performance through SFT alone. Applying RL on top of distillation provides only marginal additional gains, suggesting the traces already capture the essential reasoning patterns.
- **Orca 1 results**: A 13B Orca model matched GPT-3.5 on several reasoning benchmarks despite being roughly 10x smaller, demonstrating the power of high-quality explanation traces.
- **Orca 2 strategy selection**: Teaching small models *when* to use different reasoning strategies was critical for closing the performance gap with much larger models, enabling efficient problem-solving rather than brute-force reasoning.

## Practical Considerations

Several practical details affect the quality of reasoning distillation:

- **Trace diversity**: Generating multiple traces per problem and selecting diverse correct solutions (not just the highest-scoring one) produces more robust students. A student trained on diverse solution approaches generalizes better than one trained on a single canonical solution per problem.
- **Difficulty curriculum**: Training on traces ordered from easy to hard (curriculum learning) can improve convergence compared to random shuffling, particularly for smaller student models that struggle with difficult problems early in training.
- **Trace length management**: Very long reasoning traces (3000+ tokens) are expensive to train on and may exceed the student's effective context length. Pruning unnecessary verbosity from traces while preserving the essential reasoning steps improves training efficiency.
- **Base model selection**: The choice of base model for the student matters substantially. Models with stronger pre-training (better base capabilities) absorb reasoning traces more effectively. DeepSeek found Qwen 2.5 bases produced stronger students than Llama 3 bases at comparable parameter counts.
- **Multi-domain transfer**: Reasoning traces from one domain (mathematics) can partially transfer reasoning capability to other domains (coding, science), suggesting that some reasoning skills are domain-general.

## Common Misconceptions

- **"Distillation just teaches the student to copy the teacher's answers."** Reasoning distillation trains on the full chain-of-thought including intermediate steps, self-corrections, and strategy choices. The student learns transferable problem-solving processes that generalize to novel problems not seen during training.
- **"Smaller models cannot genuinely reason; they just memorize patterns."** The distilled models' strong performance on held-out benchmarks -- including AIME 2024 problems that postdate their training data -- demonstrates genuine generalization of reasoning capability, not pattern memorization.
- **"Direct RL is always better because the model discovers its own strategies."** At small scales, models lack the capacity for effective exploration. Distillation provides a strong foundation that RL alone cannot reach for small models. The DeepSeek-R1 findings are clear on this point.
- **"You need the teacher's logits or internal states for effective distillation."** Reasoning distillation works with only the teacher's text output (the visible reasoning traces). No access to the teacher's logits, hidden states, weights, or architecture is required, making it applicable even when the teacher is a closed-source API model.

## Connections to Other Concepts

- **Knowledge distillation**: Reasoning distillation is a specialized form of knowledge distillation focused on transferring reasoning processes rather than output probability distributions. Standard distillation uses soft logits; reasoning distillation uses explicit chain-of-thought text as the transfer medium.
- **Chain-of-thought training**: The traces used for reasoning distillation are chain-of-thought sequences. Understanding CoT prompting and its role in eliciting reasoning is essential context.
- **GRPO and RL for reasoning**: DeepSeek-R1 (the teacher) was trained with GRPO using rule-based rewards. The finding that distillation outperforms direct RL at small scales directly informs the choice between these training paradigms at different model sizes.
- **Supervised fine-tuning**: The actual training procedure for reasoning distillation is standard SFT with cross-entropy loss. The innovation lies in the training data (teacher reasoning traces), not the training algorithm.
- **Self-play and self-improvement**: Reasoning distillation is teacher-dependent (requires a stronger model), while self-improvement bootstraps from the model's own outputs. The two approaches are complementary -- a distilled model can be further improved through self-play iterations.

## Further Reading

1. **"DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning" (DeepSeek-AI, 2025, arXiv:2501.12948)** -- Presents the comprehensive reasoning distillation pipeline, the 800K trace dataset, six distilled models spanning 1.5B to 70B parameters, and the critical finding that distillation outperforms direct RL for small models.
2. **"Orca: Progressive Learning from Complex Explanation Traces of GPT-4" (Mukherjee et al., 2023, arXiv:2306.02707)** -- Pioneers the concept of explanation trace distillation from closed-source teachers, demonstrating that pedagogically rich traces produce significantly stronger students.
3. **"Orca 2: Teaching Small Language Models How to Reason" (Mitra et al., 2023, arXiv:2311.11045)** -- Advances reasoning distillation with strategy selection, teaching students not just how to reason but when to apply which reasoning approach for different problem types.
