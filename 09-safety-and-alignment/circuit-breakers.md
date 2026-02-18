# Circuit Breakers for AI Safety

**One-Line Summary**: Circuit breakers are a representation engineering-based safety mechanism where models are trained to detect harmful internal representations during generation and automatically "short-circuit" their output -- interrupting harmful completions by redirecting the model's internal states away from dangerous regions of activation space, providing a fundamentally different and more robust defense than RLHF-based refusal training.

**Prerequisites**: Understanding of representation engineering and activation steering (how behavioral concepts map to directions in activation space), transformer internals (residual stream, hidden states), RLHF and how standard safety training works, the distinction between behavioral safety (learning to refuse) and representation-level safety (modifying internal processing), familiarity with adversarial attacks and jailbreaking.

## What Are Circuit Breakers?

Standard AI safety training (RLHF, Constitutional AI) works by teaching models behavioral patterns: when the model detects a harmful request, it should produce a refusal response. This is fundamentally a pattern-matching approach -- the model learns "if input looks harmful, output a refusal." The problem is that this behavioral veneer can be bypassed by adversarial inputs that look different enough from the patterns the model was trained to refuse but still elicit harmful outputs. The knowledge remains in the model; only the refusal behavior is trained.

Circuit breakers take a fundamentally different approach. Instead of training output-level behaviors, they operate on the model's **internal representations**. The key insight: before a model produces harmful text, its internal activations must pass through "harmful" regions of representation space. If you can train the model to detect when its own internal states are entering these harmful regions and redirect those states toward safe regions, you get a defense that is much harder to bypass because it operates below the level that adversarial prompts directly manipulate.

The analogy to electrical circuit breakers is precise: just as an electrical circuit breaker detects dangerous current flow and interrupts it before damage occurs, an AI circuit breaker detects dangerous internal representations and interrupts the generation process before harmful tokens are produced.

This work was developed primarily by **Andy Zou** and collaborators at the **Center for AI Safety** and **Gray Swan AI**. Zou is a central figure in both AI safety offense (he co-developed the GCG adversarial attack) and defense (representation engineering, circuit breakers), giving him unique insight into what makes defenses robust against attacks he himself pioneered.

## How It Works

### The Core Mechanism: Representation Rerouting (RR)

The circuit breaker method, formally called **Representation Rerouting**, works as follows:

**Step 1: Define harmful and safe representation regions**

Using the representation engineering framework, collect internal activations (hidden states) from the model when processing harmful content and safe content. This produces two clusters in activation space:

- **Harmful region**: The region of activation space where internal states reside when the model is processing or about to generate harmful content.
- **Safe region**: The region where activations reside during normal, safe processing.

These regions are identified using contrastive datasets: pairs of harmful and harmless prompts and completions run through the model, with activations collected at intermediate layers.

**Step 2: Train a rerouting mechanism**

The circuit breaker training objective teaches the model to reroute its internal representations whenever they enter the harmful region. Specifically, the training loss has two components:

```
L_CB = L_reroute(harmful) + L_retain(safe)

where:
  L_reroute = ||h_l(x_harmful) - r||^2   (push harmful activations toward random/orthogonal directions)
  L_retain  = -cos_sim(h_l(x_safe), h_l_orig(x_safe))  (keep safe activations unchanged)
```

- **Rerouting loss**: When the model processes harmful inputs, its hidden states at the intervention layers are pushed toward a target that is orthogonal to (or random relative to) the harmful representation direction. The model learns to "scramble" its own internal states when it detects harmful processing.
- **Retention loss**: When the model processes safe inputs, its hidden states should remain close to the original model's hidden states. This ensures that circuit breaker training does not degrade normal performance.

**Step 3: The result at inference time**

After training, the model has internalized the circuit breaker. When given a harmful prompt (even an adversarial one), the model's internal processing automatically detects the harmful representations and reroutes them. The output that emerges is incoherent, a refusal, or simply off-topic -- the harmful generation is interrupted at the representation level.

Crucially, this happens **without explicit refusal training**. The model does not learn "detect harmful prompt, output refusal." Instead, its internal computations are disrupted in a way that prevents coherent harmful output from forming. This is why it is harder to bypass: adversarial prompts can manipulate input patterns, but they cannot directly control the model's internal representation dynamics that have been modified by circuit breaker training.

### Relationship to Representation Engineering

Circuit breakers are a direct application of representation engineering (Zou et al., 2023). The key RepE insights they build on:

1. **Harmful content has a consistent representation signature**: Across diverse types of harmful content (violence, illegal activity, dangerous knowledge), the model's internal activations pass through identifiable regions of activation space.
2. **These representations are linearly separable**: A linear classifier can distinguish harmful from safe activations with high accuracy, meaning the harmful "direction" can be identified and targeted.
3. **Representation-level interventions generalize**: Because the intervention targets the internal representation of "harmful processing" rather than specific input patterns, it generalizes to novel harmful requests -- including adversarial formulations that RLHF-trained models have never seen.

### How It Differs from RLHF-Based Safety

| Dimension | RLHF Safety | Circuit Breakers |
|-----------|------------|-----------------|
| **Level of operation** | Input-output behavior | Internal representations |
| **What is learned** | "If input pattern X, output refusal Y" | "If internal state enters region H, reroute to region S" |
| **Robustness to jailbreaks** | Brittle -- novel input patterns bypass learned refusals | More robust -- operates below the level of input manipulation |
| **Failure mode** | Model produces coherent harmful output | Model produces incoherent output (the generation breaks down) |
| **Knowledge removal** | Knowledge remains; only output behavior changes | Knowledge access is disrupted at the representation level |
| **Adversarial suffix resistance** | Low -- suffixes specifically optimized to suppress refusal behavior | Higher -- suffixes cannot directly control internal representation rerouting |
| **Computational cost** | Expensive (requires human preference data, RL training) | Moderate (requires contrastive activation data, supervised fine-tuning) |

### Training Details

**Data requirements**: Circuit breaker training requires:
- A set of harmful prompt-completion pairs (to identify harmful representations)
- A set of safe prompt-completion pairs (to define the retain set)
- These do not need to be as large or expensive as RLHF preference data

**Layers intervened on**: Circuit breakers are typically applied at multiple middle-to-late layers (e.g., layers 15-28 of a 32-layer model). These layers encode the abstract semantic content that distinguishes harmful from safe processing, while still having enough downstream computation for the rerouting to take effect.

**Training efficiency**: Circuit breaker training is a relatively lightweight fine-tuning procedure. It modifies a subset of the model's parameters through standard supervised learning with the combined rerouting and retention loss. It does not require reinforcement learning, reward models, or iterative human feedback loops.

## Why It Matters

### Robustness Against State-of-the-Art Attacks

The most significant claim about circuit breakers is their robustness against attacks that reliably break RLHF-based defenses.

In evaluations from the original paper (Zou et al., 2024), circuit breaker-equipped models showed dramatically improved robustness:

- **Against GCG adversarial suffixes**: RLHF-trained models were jailbroken 80-95% of the time. Circuit breaker models reduced this to under 5% in many configurations.
- **Against human-crafted jailbreaks**: Standard jailbreak prompts (DAN, role-playing, encoding tricks) that bypass RLHF-trained models were largely ineffective against circuit breaker-equipped models.
- **Against AutoDAN and other automated attacks**: Attack success rates dropped dramatically when targeting circuit breaker models compared to standard RLHF models.
- **Against multi-turn manipulation**: Because the circuit breaker operates on internal representations regardless of how the harmful context was established, multi-turn escalation attacks were also significantly mitigated.

### Preserving Model Utility

A key concern with any safety intervention is that it might degrade the model's usefulness on legitimate tasks. Circuit breaker evaluations showed minimal impact on standard benchmarks:

- Performance on MMLU, HellaSwag, and other general knowledge benchmarks remained within 1-2% of the original model.
- Instruction-following quality on safe prompts was largely preserved.
- The model's helpfulness ratings from human evaluators did not significantly decrease.

This is because the retention loss explicitly optimizes for preserving the model's normal activation patterns on safe inputs. The circuit breaker only activates when internal states drift toward the harmful region.

### Implications for the Safety Landscape

Circuit breakers represent a potential paradigm shift in AI safety from **behavioral safety** to **representation-level safety**:

1. **Defense in depth**: Circuit breakers can be layered on top of RLHF training, Constitutional AI, and external guardrails, providing an additional defense layer that operates at a fundamentally different level.
2. **Reducing the attack surface**: By operating on internal representations rather than input-output patterns, circuit breakers shrink the space of effective adversarial attacks. Attackers must now find inputs that not only trigger harmful outputs but also avoid triggering the circuit breaker's representation-level detection.
3. **Monitoring capability**: The same representation-level detection that triggers circuit breakers can be used for monitoring -- detecting when a model is "thinking about" harmful content even if it has not yet produced harmful output.

## Key Technical Details

- **Andy Zou's dual role**: Zou co-authored both the GCG attack paper (the most influential automated jailbreak method) and the circuit breaker defense paper. This attacker-defender dual perspective is rare and valuable in AI safety research.
- **Gray Swan AI**: The company co-founded by Zou, focused on developing robust AI safety tools. The circuit breaker work represents their core technical contribution to the field.
- **Representation rerouting targets**: The rerouted representations are pushed toward randomly initialized target vectors that are approximately orthogonal to the harmful direction. This ensures that the output after rerouting is neither harmful nor a stereotyped refusal (which could itself be gamed).
- **Connection to "refusal direction" work**: Arditi et al. (2024) showed that safety-trained models have a "refusal direction" in activation space, and that removing this direction (by subtracting it) completely eliminates refusal behavior. This finding directly motivates circuit breakers: if safety behavior is encoded as a simple direction that can be subtracted, a more robust mechanism (rerouting) is needed.
- **Multimodal applicability**: The circuit breaker framework extends naturally to multimodal models. Harmful representations from image inputs pass through the same activation space and can be detected and rerouted by the same mechanism. This is important because multimodal inputs (adversarial images) represent a growing attack surface.
- **Limitations**: Circuit breakers are not perfectly robust. Adaptive attacks that are specifically designed to avoid triggering the rerouting mechanism (by finding harmful-but-undetected regions of activation space) remain a concern. The arms race continues, but at a more favorable level for defenders.

## Common Misconceptions

**"Circuit breakers are just a better version of output filtering."** Circuit breakers operate at the representation level, not the output level. They do not examine the model's output and decide whether to block it. They modify the model's internal processing so that harmful output never coherently forms. This is a fundamentally different mechanism.

**"Circuit breakers remove harmful knowledge from the model."** Circuit breakers do not remove knowledge (that is machine unlearning). They disrupt the model's ability to access and coherently express harmful knowledge during generation. The knowledge may still exist in the weights but cannot be elicited through the normal generation process.

**"This makes models perfectly safe."** No single defense mechanism provides perfect safety. Circuit breakers significantly raise the bar for adversarial attacks but cannot guarantee that no attack will ever succeed. Adaptive adversaries will continue to search for inputs that evade detection. The goal is to make attacks dramatically harder, not impossible.

**"Circuit breakers are incompatible with RLHF."** They are complementary. The best safety profile comes from combining RLHF (for behavioral safety on common harmful patterns), circuit breakers (for representation-level robustness against novel and adversarial patterns), and external guardrails (for defense in depth).

**"The model just produces refusals more reliably."** Circuit breaker outputs on harmful inputs are typically incoherent or off-topic, not polished refusal messages. This is a feature, not a bug -- it means the defense cannot be identified and targeted through the specific form of the refusal response.

## Connections to Other Concepts

- **Representation Engineering**: Circuit breakers are a direct application of RepE principles. RepE provides the theoretical foundation (linear representation of concepts, contrastive activation analysis); circuit breakers provide a specific safety application (rerouting harmful representations).
- **Adversarial Robustness (GCG, AutoDAN)**: Circuit breakers are specifically designed to defend against the types of attacks covered in adversarial robustness research. The same researcher (Zou) developed both the leading attack (GCG) and this defense.
- **RLHF and Safety Training**: Circuit breakers address the fundamental weakness of RLHF-based safety -- that it trains behavioral patterns that can be bypassed. They provide a deeper, representation-level defense that complements behavioral training.
- **Jailbreaking**: Circuit breakers are a direct response to the jailbreaking problem. They shift the safety mechanism from a level that jailbreaks can manipulate (input-output behavior) to a level that is much harder to manipulate (internal representations).
- **Mechanistic Interpretability**: Understanding the internal representations that circuit breakers operate on requires insights from mechanistic interpretability. As our understanding of model internals improves, circuit breaker mechanisms can be made more targeted and robust.
- **Machine Unlearning**: While unlearning removes knowledge and circuit breakers disrupt access to knowledge, both aim to prevent models from producing harmful outputs. They are complementary approaches to the same goal.
- **Guardrails and Content Filtering**: Circuit breakers operate at the representation level (internal to the model), while guardrails operate at the input/output level (external to the model). Together they provide defense in depth.

## Further Reading

- Zou, A. et al. (2024). "Improving Alignment and Robustness with Circuit Breakers." *arXiv: 2406.04313.* The primary circuit breakers paper introducing representation rerouting for AI safety.
- Zou, A. et al. (2023). "Representation Engineering: A Top-Down Approach to AI Transparency." *arXiv: 2310.01405.* The foundational representation engineering framework that circuit breakers build upon.
- Arditi, A. et al. (2024). "Refusal in Language Models Is Mediated by a Single Direction." *arXiv: 2406.11717.* Demonstrates the vulnerability that motivates circuit breakers -- refusal behavior is a fragile linear feature that can be easily removed.
- Zou, A. et al. (2023). "Universal and Transferable Adversarial Attacks on Aligned Language Models." *arXiv: 2307.15043.* The GCG attack paper, co-authored by the same researcher, that defines the threat model circuit breakers defend against.
- Li, M. et al. (2024). "The Art of Defending: A Systematic Evaluation and Analysis of LLM Defense Strategies on Safety Benchmarks." Comparative analysis of various defense mechanisms including circuit breakers.
