# Knowledge Distillation

**One-Line Summary**: Knowledge distillation trains a smaller "student" model to mimic the behavior of a larger "teacher" model by learning from the teacher's soft probability distributions rather than just hard labels, transferring rich knowledge about inter-class relationships that the raw training data alone cannot convey.

**Prerequisites**: Softmax and temperature scaling, cross-entropy loss, model training basics (forward pass, backpropagation, loss functions), the concept of model capacity, fine-tuning.

## What Is Knowledge Distillation?

Consider how an experienced mentor teaches a junior colleague. The mentor does not simply say "the answer is X." They explain *why* it is X, what other possibilities were considered and why they were less likely, and what subtle patterns to look for. This nuanced guidance transfers far more knowledge than a simple answer key.

![Knowledge distillation diagram showing teacher model producing soft labels that guide student model training](https://upload.wikimedia.org/wikipedia/commons/thumb/4/48/Knowledge_Distillation.svg/800px-Knowledge_Distillation.svg.png)
*Source: [Wikimedia Commons - Knowledge Distillation](https://commons.wikimedia.org/wiki/File:Knowledge_Distillation.svg)*


Knowledge distillation works the same way. When a large teacher model predicts the next token, it does not just output the correct token -- it produces a full probability distribution over the entire vocabulary. A token predicted with 70% confidence, with 15% on a close synonym and 5% on a related word, contains rich information about the structure of language. Distillation trains the student to reproduce this full distribution, not just match the top-1 answer.

Introduced by Hinton, Vartia, and Dean in 2015, distillation has become one of the primary methods for creating smaller, deployable models from large, capable ones.

## How It Works


*Recommended visual: Illustration of soft label distributions showing dark knowledge in the teacher's probability outputs — see [Lilian Weng – The Transformer Family](https://lilianweng.github.io/posts/2023-01-27-the-transformer-family-v2/)*
*See detailed distillation architecture diagrams at: [Lilian Weng - The Transformer Family v2](https://lilianweng.github.io/posts/2023-01-27-the-transformer-family-v2/)*

### The Teacher-Student Framework

1. **Teacher model**: A large, powerful model (already trained). During distillation, its weights are frozen -- it only provides predictions.
2. **Student model**: A smaller model to be trained. It has fewer layers, smaller hidden dimensions, or both.
3. **Training data**: The same data (or a subset) used to train the teacher, or new data where the teacher generates "soft labels."

### Soft Labels vs. Hard Labels

**Hard labels** are the ground truth: the next token is "cat" (a one-hot vector with 1.0 on "cat" and 0.0 everywhere else).

**Soft labels** are the teacher's probability distribution: "cat" = 0.72, "kitten" = 0.12, "feline" = 0.05, "dog" = 0.03, ...

The soft labels encode what the teacher has learned about the relationships between tokens. The fact that "kitten" gets 12% tells the student that these words are closely related. The fact that "dog" gets 3% (not 0%) tells the student that animals are related to each other. Hard labels contain none of this information.

### Temperature in Distillation

To make the soft labels even more informative, both teacher and student outputs are passed through a temperature-scaled softmax:

$$p_i = \frac{\exp(z_i / T)}{\sum_j \exp(z_j / T)}$$

A high temperature (T = 4 to 20) softens the distributions, revealing more of the teacher's "dark knowledge" -- the subtle probability mass on tokens that would be near-zero at T = 1. This dark knowledge is where much of the useful structural information resides.

### The Distillation Loss

The total training loss is typically a weighted combination:

$$\mathcal{L} = \alpha \cdot T^2 \cdot \text{KL}\left(p_{\text{teacher}}^{(T)} \| p_{\text{student}}^{(T)}\right) + (1 - \alpha) \cdot \mathcal{L}_{\text{CE}}(y, p_{\text{student}}^{(1)})$$

Where:
- The first term is the **distillation loss**: KL divergence between the teacher's and student's softened distributions at temperature T. The T^2 factor normalizes gradients to keep them on the same scale regardless of temperature.
- The second term is the **standard cross-entropy loss** against the hard labels at temperature 1.
- Alpha controls the balance (typically 0.5-0.9, favoring the distillation loss).

### Why Soft Labels Carry More Information

Consider a vocabulary of 50,000 tokens. A hard label is a one-hot vector -- it communicates exactly log2(50,000) = ~15.6 bits of information per training example. The teacher's soft distribution communicates *much more*: it provides a probability for every token, encoding the teacher's understanding of semantic similarity, grammatical plausibility, and contextual relevance. This richer signal makes learning more sample-efficient and produces a student that generalizes better.

### Practical Examples in LLMs

**GPT-4 to smaller models**: While not publicly documented in detail, it is widely understood that many smaller commercial models benefit from distillation. A powerful model generates high-quality training data (including its reasoning traces), which is then used to train smaller models. This is sometimes called "data distillation" or "synthetic data generation," and it blurs the line between traditional distillation and training on synthetic data.

**Minitron (NVIDIA)**: The Minitron approach combines structured **pruning** with distillation:
1. Start with a large pre-trained model (e.g., 15B parameters).
2. Apply structured pruning -- remove entire attention heads, FFN neurons, or layers based on importance scores.
3. The pruned model is damaged (higher perplexity). Use the original large model as a teacher to distill knowledge back into the pruned model.
4. The result is a compact model (e.g., 8B) that performs significantly better than training an 8B model from scratch, at a fraction of the compute cost.

**DistilBERT and DistilGPT-2**: Early prominent examples where distillation produced models 40-60% smaller with 97% of the teacher's performance, running 60% faster.

### Distillation for Deployment

In production settings, distillation serves a specific role in the deployment pipeline:

*See Minitron pruning + distillation pipeline diagram at: [NVIDIA Minitron Paper (arXiv:2407.14679)](https://arxiv.org/abs/2407.14679)*


1. Train or obtain the largest, best model (teacher).
2. Evaluate what performance level is acceptable for the application.
3. Distill to the smallest student that meets the quality bar.
4. Deploy the student, which has lower latency, lower memory requirements, and lower cost per query.

This is often combined with quantization: distill to a smaller architecture, then quantize to INT4, achieving compound compression.

## Why It Matters

Distillation is the bridge between frontier model capability and practical deployment:

- **Cost reduction**: A distilled model may be 10-50x cheaper to serve per query.
- **Latency improvement**: Smaller models generate tokens faster.
- **Accessibility**: Distilled models can run on consumer hardware.
- **Specialization**: A general-purpose teacher can be distilled into a specialist student for a specific domain, often outperforming the teacher on that narrow task.

The economic significance is enormous. If a distilled 8B model can handle 90% of the queries that a 70B model handles, a serving infrastructure can route the easy queries to the cheap model and only use the expensive model for hard queries, dramatically reducing costs.

## Key Technical Details

- **Feature-based distillation**: Beyond matching output distributions, some methods match intermediate representations (hidden states, attention patterns) between teacher and student. This provides additional training signal but requires architectural compatibility.
- **Online vs. offline distillation**: Offline distillation pre-computes teacher outputs and stores them. Online distillation runs the teacher during training. Offline is more practical for large teachers but requires storage for soft labels.
- **Self-distillation**: A model distills knowledge from a larger version of itself, or from an ensemble of copies trained with different random seeds. Surprisingly effective even without a separate teacher.
- **Multi-teacher distillation**: The student learns from multiple teachers, potentially capturing diverse knowledge that no single teacher possesses.
- **Progressive distillation**: Distill in stages (e.g., 70B to 30B to 13B to 7B) rather than in one large step, which can improve final quality.

## Common Misconceptions

- **"Distillation just trains on the teacher's outputs."** While training on teacher-generated text (synthetic data) is a form of distillation, classical distillation specifically uses the full probability distributions, which contain far more information per example than the generated tokens alone.
- **"The student can match the teacher."** Generally, the student has a hard ceiling imposed by its reduced capacity. A 1B model cannot fully replicate a 70B model's capabilities. Distillation helps the student reach its potential, but that potential is bounded.
- **"Distillation is the same as fine-tuning."** Fine-tuning adapts a pre-trained model to a task using labeled data. Distillation transfers knowledge from a larger model using soft labels. They are different processes with different objectives, though they can be combined.
- **"You need the exact same training data."** The student can be distilled on different data than the teacher was trained on. In practice, using a representative dataset is sufficient.
- **"Temperature in distillation is the same as temperature in sampling."** The mathematical formula is identical, but the purpose is different. In distillation, high temperature reveals dark knowledge for training. In sampling, temperature controls output diversity.

## Connections to Other Concepts

- **Quantization**: Distillation reduces parameter count (architectural compression); quantization reduces bits per parameter (precision compression). They stack: distill, then quantize.
- **Speculative Decoding**: A distilled small model makes an excellent draft model for speculative decoding of its teacher, combining two optimizations.
- **Sampling Strategies**: The temperature parameter in distillation directly parallels temperature in sampling, though serving a different purpose.
- **Model Serving**: Distilled models are easier to serve -- they fit on fewer GPUs, have smaller KV caches, and generate tokens faster, simplifying the entire serving infrastructure.
- **Flash Attention**: Smaller distilled models benefit less from Flash Attention (shorter sequences are already fast), but the attention optimization still helps during prefill.

## Further Reading

1. **"Distilling the Knowledge in a Neural Network"** (Hinton, Vinyals, and Dean, 2015) -- The foundational paper that formalized knowledge distillation with temperature-scaled soft targets.
2. **"Compact Language Models via Pruning and Knowledge Distillation"** (Muralidharan et al., 2024) -- The Minitron paper describing NVIDIA's approach of combining structured pruning with distillation to produce efficient LLMs.
3. **"DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter"** (Sanh et al., 2019) -- A highly influential practical demonstration of distillation for transformer language models.
