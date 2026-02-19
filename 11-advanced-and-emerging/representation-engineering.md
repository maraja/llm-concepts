# Representation Engineering and Activation Steering

**One-Line Summary**: Representation engineering controls LLM behavior at inference time by identifying interpretable directions in the model's internal activation space (e.g., a "honesty direction" or "refusal direction") and adding or subtracting these steering vectors from the model's hidden states during forward passes -- modifying behavior without any weight updates or fine-tuning.

**Prerequisites**: Understanding of transformer architecture internals (residual stream, hidden states at each layer), the concept of latent representations and activation vectors, basic linear algebra (vector addition in high-dimensional spaces), and familiarity with the broader goals of AI alignment and controllability.

## What Is Representation Engineering?

Language models develop internal representations of concepts as they process text. Somewhere in the model's activations -- the high-dimensional vectors flowing through transformer layers -- there exist directions that correspond to abstract properties like honesty, harmfulness, sentiment, formality, and confidence. Representation engineering, formalized by Zou et al. (2023) at the Center for AI Safety, is the systematic study and manipulation of these internal representations.

*Recommended visual: Activation steering showing a "honesty direction" vector being added to model hidden states during inference — see [Zou et al. Representation Engineering Paper (arXiv:2310.01405)](https://arxiv.org/abs/2310.01405)*


The core discovery: many behavioral properties of LLMs are encoded as roughly linear directions in activation space. If you can identify the "honesty direction" -- a vector in the model's hidden state space that separates honest from dishonest activations -- you can add this vector to the model's activations during generation to make it more honest, or subtract it to make it less honest. This works because the model's internal geometry is surprisingly linear for many high-level concepts.

Activation steering (a closely related technique, also called "inference-time intervention" or "representation reading and control") is the practical application: given a steering vector for a desired property, add a scaled version of that vector to the model's residual stream at a specific layer during every forward pass. The model's behavior shifts in the direction of the concept encoded by the vector, without changing any model weights.

## How It Works


*Recommended visual: Refusal direction in activation space showing how a single direction mediates safety refusal behavior — see [Arditi et al. Refusal in Language Models (arXiv:2406.11717)](https://arxiv.org/abs/2406.11717)*

### Finding Steering Vectors: The Contrastive Approach

The most common method for identifying a steering vector for a concept (e.g., "honesty") follows a contrastive pattern:

**Step 1: Create contrastive pairs**

Design pairs of prompts that differ primarily in the target concept:

```
Positive (honest): "Pretend you are an honest person. [Question]"
Negative (dishonest): "Pretend you are a dishonest person. [Question]"
```

Or more sophisticated paired datasets:
```
Positive: "The capital of France is Paris." (true statement)
Negative: "The capital of France is London." (false statement)
```

**Step 2: Collect activations**

Run both sets of prompts through the model and extract the hidden state activations at a specific layer (or across multiple layers). For a model with hidden dimension d, each prompt produces a d-dimensional activation vector at each token position (typically using the last token or a mean across token positions).

**Step 3: Compute the steering vector**

The steering vector is the difference between the mean positive and mean negative activations:

```
v_steer = mean(activations_positive) - mean(activations_negative)
```

This difference vector captures the "direction" in activation space that the model associates with the target concept. More sophisticated methods use PCA (Principal Component Analysis) on the difference vectors to find the primary direction, which can be more robust when using many contrastive pairs.

**Step 4: Validate**

Test the steering vector by projecting held-out activations onto it. If the vector genuinely captures the target concept, positive examples should have high dot products with the vector and negative examples should have low dot products. This is "representation reading" -- using the vector as a probe to read the model's internal state.

### Applying Steering Vectors

Once you have a steering vector v_steer, you modify the model's behavior by adding a scaled version to the residual stream during inference:

```
h'_l = h_l + alpha * v_steer
```

Where:
- `h_l` is the original hidden state at layer l
- `alpha` is a scaling coefficient that controls the strength of the intervention
- `h'_l` is the modified hidden state that continues through the rest of the model

**Layer selection**: The intervention is typically applied at a specific layer or range of layers. Middle layers (e.g., layers 15-25 of a 32-layer model) often work best because:
- Early layers process low-level features (syntax, basic semantics) and are less concept-rich.
- Later layers are close to the output and may be too specific, leaving less room for the intervention to propagate through subsequent computation.
- Middle layers represent abstract concepts while still having enough downstream computation for the intervention to affect output.

**Scaling coefficient (alpha)**: This controls the intervention strength.
- Small alpha (0.5-2.0): Subtle behavioral nudges. The model's output quality is preserved but behavior shifts slightly.
- Medium alpha (2.0-5.0): Noticeable behavioral changes while maintaining coherence.
- Large alpha (> 5.0): Dramatic behavioral changes, but risks incoherent or degenerate output.

The optimal alpha is empirically determined and depends on the model, the concept, and the desired strength of the effect.

### Key Variations

**Refusal steering**: Turner et al. (2024) at Anthropic demonstrated that refusal behavior in Claude and similar models can be controlled by activation steering. A "refusal direction" was identified such that adding it makes the model more likely to refuse requests, and subtracting it makes the model more likely to comply. This finding has implications for both safety (can we make models more reliably refuse harmful requests?) and adversarial attacks (can steering vectors bypass safety training?).

**Representation reading (RepReading)**: Beyond steering, the same vectors can be used as linear probes to read the model's internal state. By projecting activations onto the honesty vector, you can estimate whether the model "believes" its own output to be truthful -- even before the output is generated. This has applications in hallucination detection and confidence calibration.

**Multi-concept steering**: Multiple steering vectors can be applied simultaneously (e.g., increase honesty AND decrease verbosity). Because the vectors are in a high-dimensional space, orthogonal concepts do not interfere with each other. Non-orthogonal concepts may interact, requiring careful balancing of alpha values.

**Function vectors**: Todd et al. (2024) showed that specific input-output functions (like "translate to French" or "convert to uppercase") are also encoded as directions in activation space. Adding the corresponding vector causes the model to perform that function, even without explicit instructions in the prompt.

### The Linear Representation Hypothesis

Representation engineering rests on the empirical observation that many concepts are represented as linear directions in LLM activation spaces. This "linear representation hypothesis" has been validated across multiple models and concepts:

- Honesty/deception
- Positive/negative sentiment
- Formal/informal register
- Confident/uncertain
- Harmful/harmless
- Sycophantic/non-sycophantic
- Various emotions (happiness, anger, fear)

The hypothesis is not that all concepts are linear -- complex, compositional concepts likely have more complex geometries. But for many behaviorally relevant properties, linearity holds well enough for practical steering.

Why linearity? One explanation is that during training, the model learns to linearly separate concepts that are relevant to next-token prediction. Linear separation is the simplest geometric structure that enables downstream layers to easily compose and use these features.

## Why It Matters

Representation engineering matters for several reasons that span safety, controllability, and interpretability:

**Lightweight behavioral control**: Fine-tuning (RLHF, DPO) requires training data, compute, and permanent weight changes. Activation steering achieves behavioral modification at inference time with zero training cost. You can turn a steering vector on and off dynamically, adjust its strength per-request, or apply different steering vectors for different users or contexts.

**Safety and alignment**: If we can identify and manipulate the internal representations of honesty, harmfulness, and compliance, we can potentially build more robust safety mechanisms that operate at the representation level rather than relying solely on training-time interventions (which can be fragile).

**Interpretability**: Representation engineering provides a window into what the model "knows" and "believes" internally. Being able to read the model's internal state for honesty, confidence, and other properties is a step toward understanding and monitoring AI systems.

**Adversarial implications**: The ability to remove safety behaviors through steering vectors (as demonstrated in several papers) reveals that training-time safety measures may be more brittle than assumed. This is concerning but also informative for building more robust defenses.

**Practical applications**: Deployment-time tuning of model personality, formality, creativity, and other properties without maintaining multiple fine-tuned model variants. A single base model with a library of steering vectors can serve diverse use cases.

## Key Technical Details

- **Computational overhead**: Adding a vector to hidden states is nearly free computationally -- it is a single vector addition per layer per token. The overhead is negligible compared to the transformer forward pass.
- **Storage**: A steering vector is a single d-dimensional vector (e.g., 4096 floats for a 7B model = 16KB). A library of 100 steering vectors requires ~1.6MB. This is vastly more storage-efficient than maintaining multiple fine-tuned models.
- **Layer sensitivity**: The choice of which layer(s) to intervene at significantly affects both the strength and the nature of the behavioral change. Some concepts are better controlled at earlier layers, others at later layers. Systematic sweeps across layers are standard practice.
- **Token position**: Interventions can be applied at all token positions or only at specific positions (e.g., only at the last token of the prompt, or only at generation tokens). The choice affects the nature and stability of the behavioral change.
- **Normalization**: Some implementations normalize the steering vector before scaling, so that alpha directly controls the magnitude relative to the model's typical activation norms. Without normalization, the optimal alpha varies across models and layers.
- **Contrastive pair quality**: The quality of the steering vector depends critically on the quality and diversity of the contrastive pairs. Pairs that differ in multiple concepts (not just the target concept) produce impure steering vectors that have unintended side effects.

## Common Misconceptions

**"Representation engineering is the same as fine-tuning."** They are fundamentally different. Fine-tuning modifies model weights permanently through gradient descent. Representation engineering modifies activations at inference time by vector addition. No weights are changed. The intervention is reversible, adjustable, and composable.

**"Steering vectors are like prompt engineering."** While both modify behavior without changing weights, they operate at entirely different levels. Prompt engineering modifies the model's input; steering vectors modify the model's internal representations directly. Steering vectors can induce behavioral changes that are difficult or impossible to achieve through prompting alone, and they work even when the model's input processing might resist prompt-based instructions.

**"All concepts are linearly represented."** The linear representation hypothesis holds well for many high-level behavioral properties, but it is not universal. Some concepts may have nonlinear or polysemantic representations. Compositional concepts (e.g., "dishonest AND formal") may require more sophisticated intervention strategies.

**"Activation steering is robust and predictable."** Current steering techniques can have side effects. Steering for honesty might also affect the model's confidence calibration, verbosity, or topic coverage in unexpected ways. The interactions between concepts in activation space are not fully understood.

**"This can replace RLHF for alignment."** Representation engineering is a complementary tool, not a replacement. RLHF shapes the model's default behavior through training; steering vectors provide fine-grained, dynamic control on top of that trained behavior. A model that has been poorly aligned through training is unlikely to be fully correctable through steering alone.

## Connections to Other Concepts

- **Mechanistic interpretability**: Representation engineering is closely related to the broader field of mechanistic interpretability. While mechanistic interpretability aims to understand the computational mechanisms within neural networks (circuits, features, superposition), representation engineering focuses specifically on reading and controlling high-level behavioral properties through their geometric representation in activation space.
- **RLHF and alignment**: Representation engineering provides an alternative/complementary control mechanism to training-time alignment techniques. Understanding how alignment training modifies internal representations is an active research question.
- **Guardrails**: Steering vectors could serve as a representation-level guardrail -- detecting and intervening on harmful content at the activation level rather than at the input/output level.
- **AI safety**: The ability to read the model's internal state (e.g., detecting deception by reading the "honesty direction") is directly relevant to scalable oversight and AI safety monitoring.
- **LoRA and PEFT**: Like LoRA, representation engineering provides a parameter-efficient way to modify model behavior. LoRA adds low-rank weight updates; steering adds activation-level vectors. LoRA requires training; steering does not.
- **Probing classifiers**: Linear probes (training a linear classifier on activations to predict a property) are the precursor to representation engineering. Probes read properties; steering vectors both read and control them.

## Further Reading

- Zou, A. et al. (2023). "Representation Engineering: A Top-Down Approach to AI Transparency." (arXiv: 2310.01405) The foundational paper from the Center for AI Safety, introducing representation reading and control across multiple concepts and models.
- Turner, A. et al. (2024). "Activation Addition: Steering Language Models Without Optimization." (arXiv: 2308.10248) Demonstrates activation steering for controlling model behavior, with detailed analysis of refusal, sentiment, and other properties.
- Li, K. et al. (2024). "Inference-Time Intervention: Eliciting Truthful Answers from a Language Model." *NeurIPS 2024.* (arXiv: 2306.03341) Identifies "truthfulness directions" and uses them to make models more truthful during generation.
- Todd, E. et al. (2024). "Function Vectors in Large Language Models." *ICLR 2024.* Demonstrates that input-output functions (not just behavioral properties) are encoded as linear directions.
- Templeton, A. et al. (2024). "Scaling Monosemanticity: Extracting Interpretable Features from Claude 3 Sonnet." *Anthropic Research.* Related work on extracting interpretable features from LLM activations, providing a foundation for targeted steering.
