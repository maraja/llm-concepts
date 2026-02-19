# Full Fine-Tuning vs PEFT: When to Use What

**One-Line Summary**: Full fine-tuning updates every parameter in a model for maximum adaptability but at enormous compute and memory cost, while PEFT methods achieve surprisingly competitive quality by training only a small fraction of parameters -- and at sufficient model scale, the gap between them effectively vanishes.

**Prerequisites**: Understanding of what fine-tuning means (continuing training on task-specific data), familiarity with at least one PEFT method (LoRA recommended), basic awareness of GPU memory constraints, and the concepts of catastrophic forgetting and overfitting.

## What Is This Comparison About?

Imagine renovating a house. Full fine-tuning is a complete gut renovation -- you tear out every wall, rewire the electrical, replace the plumbing, and rebuild from the studs up. You get exactly the house you want, but it costs a fortune, takes months, and requires a large crew. PEFT methods are targeted renovations -- you repaint the walls, update the fixtures, and remodel the kitchen. The house looks and functions remarkably differently, but the foundation, framing, and most infrastructure remain untouched. For most purposes, the targeted renovation achieves 95% of the full renovation's outcome at 5% of the cost.

*Recommended visual: Parameter count comparison across PEFT methods showing trainable vs frozen parameters — see [Hugging Face PEFT Documentation](https://huggingface.co/docs/peft/index)*


This comparison is about understanding when you truly need the gut renovation (full fine-tuning) versus when the targeted approach (PEFT) is not just cheaper but actually preferable.

## How It Works


*Recommended visual: Performance vs parameter efficiency trade-off curves for full fine-tuning, LoRA, QLoRA, and adapters — see [Scaling Down to Scale Up: A Guide to Parameter-Efficient Fine-Tuning (arXiv:2303.15647)](https://arxiv.org/abs/2303.15647)*

### Full Fine-Tuning

In full fine-tuning, every parameter in the pretrained model is unfrozen and updated via gradient descent on a task-specific dataset:

```
theta_new = theta_pretrained - learning_rate * gradient(Loss(theta, D_task))
```

where theta represents all model parameters and D_task is the task dataset.

**Memory requirements** for full fine-tuning of a model with N parameters:
- Model weights: 2N bytes (in fp16/bf16)
- Gradients: 2N bytes (in fp16/bf16)
- Optimizer states (Adam): 4N bytes for momentum + 4N bytes for variance (in fp32, as required for numerical stability)
- Activations: Variable, but often comparable to model size with typical batch sizes
- **Total**: Approximately 12-16N bytes minimum

For a 70B parameter model: 12 x 70B = 840 GB minimum -- requiring a cluster of 8-16 high-end GPUs.

**Training characteristics**:
- Learning rate must be small (typically 1e-5 to 5e-5) to avoid catastrophic forgetting of pretrained knowledge
- Full gradient computation through all layers
- All parameters participate in the optimization landscape
- Risk of overfitting is high on small datasets because of the vast parameter count

### PEFT Fine-Tuning (Using LoRA as Representative)

In PEFT, the pretrained parameters theta_0 are frozen and a small set of adapter parameters phi are trained:

```
phi_new = phi_init - learning_rate * gradient(Loss(theta_0, phi, D_task))
```

**Memory requirements** for LoRA fine-tuning (rank 16, all attention layers):
- Model weights: 2N bytes (frozen, no gradients needed; can be quantized to 0.5N with QLoRA)
- LoRA parameters: ~0.01-0.1% of N, negligible
- LoRA gradients: Same as LoRA parameters, negligible
- LoRA optimizer states: ~2-4x LoRA parameters, negligible
- Activations: Similar to full fine-tuning (this is often the overlooked cost)
- **Total**: Approximately 2N bytes (or 0.5N with QLoRA), plus activation memory

For a 70B parameter model with QLoRA: ~35 GB for weights + activations -- feasible on a single 80GB GPU.

**Training characteristics**:
- Learning rate can be larger (1e-4 to 3e-4) because the adaptation is structurally constrained
- Gradients only flow through the small adapter parameters (though activations are still computed through the full model)
- The low-rank constraint acts as an implicit regularizer, reducing overfitting risk
- Catastrophic forgetting is structurally mitigated because pretrained weights are untouched

## Why It Matters

The choice between full fine-tuning and PEFT is one of the most consequential practical decisions in applied LLM work. It determines:

- **Infrastructure costs**: Full fine-tuning a 70B model can cost $10,000-50,000+ in cloud compute. QLoRA fine-tuning the same model might cost $100-500.
- **Iteration speed**: A LoRA fine-tuning run might take 2-4 hours. Full fine-tuning the same model could take 2-4 days.
- **Deployment architecture**: PEFT enables multi-tenant serving (one base model, many adapters). Full fine-tuning produces entirely separate model copies.
- **Team accessibility**: PEFT puts model customization in the hands of small teams and individual researchers. Full fine-tuning remains the domain of well-funded organizations.

### The Convergence at Scale

Perhaps the most important finding in PEFT research is that **as model scale increases, the gap between PEFT and full fine-tuning shrinks**. This has been demonstrated across multiple studies:

- Lester et al. (2021) showed prompt tuning (the simplest PEFT method) matches full fine-tuning at 10B+ parameters.
- Hu et al. (2021) showed LoRA matches full fine-tuning on GPT-3 175B for many tasks.
- Dettmers et al. (2023) showed QLoRA fine-tuning of a 65B model matches full-precision fine-tuning.

The intuition: larger models have learned richer, more general representations during pretraining. Adapting these representations to a new task genuinely requires only a low-dimensional adjustment. Smaller models, with their less complete representations, benefit more from the full flexibility of updating all parameters.

## Key Technical Details

### When Full Fine-Tuning Is Necessary

- **Major domain shifts**: If the target domain is radically different from the pretraining data (e.g., adapting an English-only model to a low-resource language, or adapting a text model to code), full fine-tuning may capture adaptations that PEFT cannot.
- **Pretraining continuation**: When the goal is to extend the model's foundational knowledge (sometimes called continued pretraining or domain-adaptive pretraining), full fine-tuning is standard because the changes are distributed across all layers.
- **Maximum quality, no cost constraint**: When you are training a flagship model and every 0.1% of benchmark performance matters (e.g., Llama, Mistral, or other foundation model providers doing post-training).
- **Small models**: For models under ~1B parameters, PEFT methods leave significant quality on the table. The structural constraint of low-rank updates is more restrictive when the model's representations are less rich.
- **Architecture modifications**: If you are adding new token types, changing the vocabulary, or modifying the architecture, full fine-tuning (or at least training a larger subset of parameters) is typically required.

### When PEFT Is Better

- **Limited compute budget**: When GPU hours are constrained, PEFT lets you fine-tune larger, more capable models rather than fully fine-tuning smaller ones. A QLoRA-tuned 70B model typically outperforms a fully fine-tuned 7B model.
- **Multi-tenant serving**: When you need to serve many task-specific variants from a single base model, LoRA adapters (10-50 MB each) are orders of magnitude more efficient than separate full model copies (tens of GB each).
- **Rapid experimentation**: When testing many hypotheses about data, tasks, or hyperparameters, PEFT's fast iteration cycle is invaluable.
- **Small datasets**: With fewer than 10,000-50,000 training examples, PEFT's implicit regularization (from the low-rank constraint) often produces better results than full fine-tuning, which tends to overfit.
- **Preserving base capabilities**: PEFT structurally prevents catastrophic forgetting. If you need the model to retain strong general-purpose abilities while adding a specialized skill, PEFT is safer.
- **Composability**: You may want to combine adapters (e.g., one for code, one for medical knowledge) or dynamically switch between them.

### Practical Decision Framework

```
START
  |
  v
Is your model > 10B parameters?
  |-- Yes --> PEFT will likely match full fine-tuning quality
  |           |
  |           v
  |           Use LoRA/QLoRA unless you have specific
  |           reasons for full fine-tuning
  |
  |-- No --> Is your dataset > 100K examples and
  |          domain shift is large?
              |-- Yes --> Consider full fine-tuning
              |-- No --> PEFT is still likely better
                         (regularization benefit)

Additional factors:
- Budget < $1,000? --> QLoRA
- Need multi-tenant serving? --> LoRA (mandatory)
- Continued pretraining? --> Full fine-tuning
- Dataset < 10K examples? --> PEFT (avoid overfitting)
- Need to preserve base capabilities? --> PEFT
```

### Hybrid Approaches

The binary choice between full and PEFT is increasingly blurred:

- **Selective unfreezing**: Freeze most layers but fully fine-tune the top few layers plus LoRA on the rest.
- **Progressive unfreezing**: Start with PEFT, then gradually unfreeze layers if quality is insufficient.
- **Layer-wise LoRA rank**: Use higher ranks for layers that need more adaptation (often the later layers) and lower ranks for earlier layers.
- **Training embedding and LM head**: Even when using LoRA for attention layers, it is common to fully train the embedding layer and language modeling head, especially when the vocabulary has been modified.

## Common Misconceptions

- **"Full fine-tuning always produces better models."** On small datasets (under ~50K examples), PEFT frequently outperforms full fine-tuning due to its regularizing effect. On large models (10B+), the gap is often negligible even on large datasets.
- **"PEFT is just for people who cannot afford full fine-tuning."** PEFT offers genuine advantages beyond cost: multi-tenant serving, composability, reduced catastrophic forgetting, and faster experimentation. Many organizations that could afford full fine-tuning choose PEFT for these reasons.
- **"You should always use the largest rank possible for LoRA."** There is an optimal rank for each task. Beyond that point, increasing rank wastes parameters without improving quality and can even hurt performance through overfitting.
- **"PEFT methods are interchangeable."** Different PEFT methods have substantially different quality profiles, parameter counts, and inference characteristics. The choice of method matters.
- **"Full fine-tuning is obsolete."** For continued pretraining, very large domain shifts, small models, and flagship model training, full fine-tuning remains the correct choice.

## Connections to Other Concepts

- **LoRA and QLoRA**: The most practical PEFT methods and the default recommendation for most fine-tuning scenarios.
- **Distributed training**: Full fine-tuning of large models requires distributed training (FSDP, DeepSpeed). PEFT often eliminates this requirement, simplifying the infrastructure.
- **Catastrophic forgetting**: Full fine-tuning risks overwriting pretrained knowledge. PEFT structurally mitigates this by keeping pretrained weights frozen.
- **Overfitting and regularization**: PEFT's low-rank constraint acts as implicit regularization. Full fine-tuning may require explicit regularization (weight decay, dropout, early stopping) to avoid overfitting on small datasets.
- **Inference optimization**: Merged LoRA adapters have zero inference overhead, making them compatible with all inference optimizations (quantization, KV caching, speculative decoding). Separate adapter architectures may complicate some optimizations.
- **Model merging and task arithmetic**: PEFT adapters enable model merging techniques (e.g., TIES merging, DARE) that combine capabilities from multiple fine-tuned variants without additional training.
- **Instruction tuning and RLHF**: Both the supervised fine-tuning (SFT) stage and the reinforcement learning stage of alignment can use either full fine-tuning or PEFT. Many open-source alignment efforts use LoRA for SFT and full fine-tuning (or LoRA) for the reward model.

## Further Reading

- **"Scaling Down to Scale Up: A Guide to Parameter-Efficient Fine-Tuning"** -- Lialin et al. (2023). A comprehensive survey covering the full PEFT landscape with systematic comparisons. [arXiv:2303.15647](https://arxiv.org/abs/2303.15647)
- **"LoRA: Low-Rank Adaptation of Large Language Models"** -- Hu et al. (2021). Section 4 contains direct comparisons between LoRA and full fine-tuning across multiple model scales. [arXiv:2106.09685](https://arxiv.org/abs/2106.09685)
- **"Full Parameter Fine-Tuning for Large Language Models with Limited Resources"** -- Liao et al. (2024). Explores techniques to make full fine-tuning more accessible, including memory-efficient optimizers and gradient accumulation strategies. [arXiv:2306.09782](https://arxiv.org/abs/2306.09782)
