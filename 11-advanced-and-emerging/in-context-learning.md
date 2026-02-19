# In-Context Learning

**One-Line Summary**: In-context learning (ICL) is the emergent ability of large language models to learn new tasks from examples provided in the prompt at inference time, without any gradient updates or parameter changes.

**Prerequisites**: Transformer architecture, attention mechanisms, pre-training, tokenization, prompt engineering

## What Is In-Context Learning?

Imagine you hire a new employee and, instead of sending them through a week-long training program, you simply show them three completed examples of the work you need done and say, "Now do this one." Remarkably, they produce the correct output. That is in-context learning: a model that was trained once on a massive corpus can pick up entirely new tasks on the fly, just from a handful of demonstrations placed in its input.

*Recommended visual: Few-shot in-context learning example showing demonstrations in the prompt enabling task performance without gradient updates — see [Lilian Weng – Prompt Engineering](https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/)*


In-context learning is one of the most surprising emergent capabilities of large language models. A single pretrained model -- with completely frozen parameters -- can perform translation, sentiment classification, code generation, mathematical reasoning, data reformatting, and thousands of other tasks simply by varying the prompt. No separate fine-tuning run, no task-specific head, no additional training cost. You literally describe or demonstrate what you want, and the model does it.

The phenomenon was first prominently demonstrated in GPT-3 (Brown et al., 2020), where the authors showed that scaling model size dramatically improved few-shot performance. A 175B parameter model could perform tasks it had never been explicitly trained on, given just a few input-output pairs in the prompt. This discovery fundamentally shifted the paradigm from "one model per task" to "one model, many tasks."

What makes ICL particularly fascinating to researchers is that it was not explicitly designed or trained for. No one programmed transformers to learn from demonstrations -- the capability emerged spontaneously from the combination of self-attention, massive scale, and diverse pre-training data. Understanding why this happens has become one of the central open questions in LLM research.

## How It Works


*Recommended visual: In-context learning performance scaling with number of demonstrations and model size — see [GPT-3 Paper (arXiv:2005.14165)](https://arxiv.org/abs/2005.14165)*

### The Mechanics of Prompting

In-context learning typically takes one of three forms, defined by how many demonstrations are provided:

- **Zero-shot**: No examples, just a task instruction. E.g., `"Translate the following English sentence to French: 'The cat sat on the mat.'"`
- **Few-shot**: A small number (typically 1-8) of input-output pairs followed by a new input.
- **Many-shot**: Dozens or even hundreds of examples, enabled by long-context models.

A typical few-shot prompt looks like this:

```
Classify the sentiment of each review as Positive or Negative.

Review: "The food was absolutely delicious and the service was impeccable."
Sentiment: Positive

Review: "Waited 45 minutes for cold, bland pasta. Never again."
Sentiment: Negative

Review: "A cozy atmosphere with creative cocktails and friendly staff."
Sentiment:
```

The model completes the final line with "Positive" -- it has "learned" the classification task from two examples, despite never being fine-tuned for sentiment analysis.

### Sensitivity to Demonstration Design

Research has uncovered several surprising findings about what makes ICL demonstrations effective:

**Format over labels**: Min et al. (2022) showed that the correctness of labels in demonstrations matters less than the format and input distribution. Even randomly labeled examples can improve ICL performance over zero-shot, because they convey the input space, label space, and output format. This finding suggests that ICL partly works by activating a latent task template rather than learning a mapping from the examples.

**Example selection**: The choice of which examples to include dramatically affects performance. Retrieval-augmented example selection -- choosing demonstrations that are semantically similar to the test input -- can improve accuracy by 10-20% over random selection. Conversely, adversarially chosen examples can reduce accuracy below zero-shot levels.

**Example ordering**: The sequence in which demonstrations appear affects ICL performance, sometimes by 10-30 percentage points. Models exhibit recency bias (overweighting later examples) and primacy effects, suggesting that positional attention patterns interact with the learning mechanism.

### Theoretical Explanations

Why does ICL work? Several competing (and possibly complementary) theories have emerged:

**Implicit Bayesian Inference**: The model has seen many tasks during pre-training. Given demonstrations, it performs approximate Bayesian inference to identify which task distribution the examples came from, then applies that task to the new input. The demonstrations narrow the posterior over possible tasks.

**Implicit Gradient Descent**: Akyurek et al. (2023) and Von Oswald et al. (2023) showed that transformer attention layers can implement something functionally equivalent to gradient descent. The forward pass through attention on the demonstration examples produces effects similar to training on those examples -- the model effectively "trains itself" within a single forward pass.

**Induction Heads**: Olsson et al. (2022) identified specific attention head circuits called "induction heads" that perform in-context pattern matching. These heads implement the rule: "if pattern A was followed by B earlier in the context, then when A appears again, predict B." This simple mechanism, operating across two attention layers, forms a primitive but powerful pattern-completion engine that underlies much of ICL.

**Task Vector Formation**: Hendel et al. (2023) demonstrated that demonstrations create an internal "task vector" -- a direction in activation space that encodes what task is being performed. This vector modifies how subsequent inputs are processed, effectively steering the model toward the correct task behavior.

## Why It Matters

1. **Eliminates per-task fine-tuning**: A single model deployment can handle thousands of tasks, drastically reducing engineering overhead, compute costs, and deployment complexity.
2. **Enables rapid prototyping**: Developers can test whether a model can perform a new task in seconds by writing a prompt, rather than spending hours or days assembling training data and running fine-tuning jobs.
3. **Democratizes AI capabilities**: Users without ML expertise can leverage powerful models by simply describing tasks in natural language with examples.
4. **Reveals deep properties of transformers**: ICL demonstrates that transformers are not merely pattern memorizers -- they learn general-purpose learning algorithms during pre-training that can be applied to novel tasks at inference time.
5. **Drives practical applications**: ICL is the foundation of virtually all prompt-based applications, from chatbots to code assistants to data processing pipelines.

## Key Technical Details

- ICL performance scales with model size: GPT-3's 175B model dramatically outperformed its 1.3B variant on few-shot tasks, suggesting ICL is an emergent capability that appears above certain scale thresholds.
- Example ordering matters significantly -- the same examples in different orders can change accuracy by 10-30 percentage points on classification tasks.
- Label space and format consistency across demonstrations is critical; mismatched formats degrade performance.
- ICL does not update model weights; all "learning" happens through the attention mechanism operating over the extended context.
- Performance generally improves with more demonstrations up to a point (typically 4-16 examples), then plateaus or can even degrade as the context fills with redundant information.
- Models can exhibit "task ambiguity" when demonstrations are consistent with multiple tasks -- providing more diverse examples helps disambiguate.
- ICL can compose multiple skills: a prompt can simultaneously demonstrate formatting, domain knowledge, and reasoning style, and the model integrates all of these into its output.
- Retrieval-augmented few-shot selection (choosing demonstrations most similar to the test input) can significantly boost ICL performance over random example selection.
- Many-shot ICL (32-256+ examples) enabled by long-context models can approach fine-tuning-level performance on certain tasks, blurring the line between prompting and training.
- The "task vector" created by ICL demonstrations can be extracted and applied to new inputs without the original demonstrations, confirming that ICL creates a transferable internal representation.
- ICL quality is highly sensitive to demonstration diversity: examples covering edge cases and different subcategories of the task improve robustness far more than repeated similar examples.
- Models exhibit "majority label bias" in few-shot classification: if 3 of 4 demonstrations have the same label, the model disproportionately predicts that label regardless of the test input. Balanced label representation in demonstrations mitigates this.

## Common Misconceptions

- **"In-context learning actually updates the model's weights."** It does not. The parameters remain completely frozen. All adaptation happens through the forward pass of the attention mechanism operating over the concatenated demonstrations and query. This is what makes ICL so remarkable -- and so different from fine-tuning.

- **"More examples always means better performance."** There are diminishing returns, and in some cases performance can degrade with too many examples due to context window limitations, attention dilution, or the introduction of noisy/contradictory demonstrations. Quality and diversity of examples often matter more than quantity.

- **"ICL is just memorization of training data patterns."** While pre-training exposure helps, models can perform ICL on genuinely novel task formats they have never seen during training. The implicit gradient descent theory suggests the model has learned a general-purpose learning algorithm, not just specific task templates.

- **"Zero-shot and few-shot are fundamentally different mechanisms."** They likely exist on a continuum. Zero-shot relies on the instruction activating a latent task representation; few-shot provides additional signal that sharpens and disambiguates that representation.

- **"ICL performance is deterministic for a given prompt."** Due to the sensitivity to example ordering, selection, and formatting, small changes to a prompt can produce significantly different results. This variability means that ICL-based evaluations should average over multiple prompt configurations to get reliable performance estimates.

## Connections to Other Concepts

- **Prompt Engineering**: ICL is the mechanism that makes prompt engineering effective -- designing prompts is really designing learning curricula for the model's in-context learner.
- **Fine-Tuning / LoRA**: The alternative to ICL for task adaptation. Fine-tuning modifies weights permanently and can achieve higher ceiling performance, but at greater cost and with less flexibility.
- **Chain-of-Thought Prompting**: A specialized form of ICL where the demonstrations include reasoning steps, leveraging ICL to teach the model not just what to output but how to think.
- **Attention Mechanisms**: The self-attention mechanism is the computational substrate that enables ICL -- particularly multi-head attention's ability to perform flexible pattern matching across the context window.
- **Emergent Capabilities**: ICL is one of the canonical examples of an emergent capability -- an ability that appears qualitatively only above certain model scale thresholds.
- **Retrieval-Augmented Generation (RAG)**: RAG can be viewed as a form of dynamic ICL, where retrieved documents serve as demonstrations that ground the model's responses in specific evidence.
- **Context Window Length**: The practical limit of ICL is determined by the model's context window. Longer contexts enable many-shot ICL, which approaches fine-tuning-level performance on some tasks.
- **Scaling Laws**: ICL performance improves predictably with model scale, and the gap between few-shot and fine-tuned performance narrows as models grow larger, suggesting that sufficiently large models may close this gap entirely.

## Further Reading

- Brown et al., "Language Models are Few-Shot Learners" (2020) -- the GPT-3 paper that demonstrated ICL at scale
- Olsson et al., "In-context Learning and Induction Heads" (2022) -- mechanistic analysis identifying induction heads as a key ICL circuit
- Akyurek et al., "What Learning Algorithm is In-Context Learning?" (2023) -- formal connection between ICL and gradient descent
- Min et al., "Rethinking the Role of Demonstrations" (2022) -- surprising finding that label correctness matters less than format and input distribution in ICL
- Von Oswald et al., "Transformers Learn In-Context by Gradient Descent" (2023) -- theoretical and empirical evidence for the implicit gradient descent view
- Hendel et al., "In-Context Learning Creates Task Vectors" (2023) -- demonstrates that ICL creates extractable task representations in activation space
- Wei et al., "Larger Language Models Do In-Context Learning Differently" (2023) -- shows how ICL behavior changes qualitatively with scale, including the ability to override semantic priors
