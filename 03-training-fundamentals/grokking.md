# Grokking

**One-Line Summary**: Grokking is the phenomenon where a neural network suddenly generalizes to unseen data long after it has already memorized the training set, challenging assumptions about when and how models truly learn.

**Prerequisites**: Training and validation loss, overfitting, regularization, gradient descent, early stopping

## What Is Grokking?

Imagine a student preparing for a math exam by memorizing every answer in the practice workbook. For weeks, they can recite solutions perfectly but fail every new problem the teacher throws at them. Then one morning -- seemingly out of nowhere -- something clicks. The student suddenly understands the underlying rules and can solve any problem of that type, even ones they have never seen. That abrupt transition from rote memorization to genuine understanding is grokking.

In neural network training, grokking manifests as a dramatic phase transition. The model quickly reaches near-perfect training accuracy (memorization), while validation accuracy remains at chance level. Training continues for thousands or even millions of additional steps with no visible improvement on the validation set. Then, abruptly, validation accuracy shoots up to match training accuracy. The model has "grokked" the task -- it has discovered the generalizable structure hidden beneath the memorized solutions.

This phenomenon was first documented by Power et al. (2022) at OpenAI, who observed it on simple algorithmic tasks like modular arithmetic (e.g., learning (a + b) mod 97). The model would memorize all training pairs quickly, then continue training for 100x longer before suddenly generalizing. The discovery was startling because it directly contradicts the standard practice of early stopping, which would have terminated training long before generalization occurred.

The name "grokking" comes from Robert Heinlein's science fiction novel "Stranger in a Strange Land" (1961), where it means to understand something so thoroughly and completely that it becomes part of you. The term was adopted in hacker culture and fits perfectly here: the model does not merely fit the data -- it eventually groks the underlying structure, transitioning from surface-level memorization to deep comprehension.

## How It Works

### The Two Phases of Learning

Grokking reveals that training proceeds in two distinct phases with a long gap between them:

**Phase 1 -- Memorization (fast)**: The model rapidly fits the training data by learning a lookup table. Each training example is stored almost independently. This is computationally easy -- the model just needs enough parameters to memorize the dataset. Training loss drops to near zero. Validation loss remains high.

```
Epoch 100:   Train Acc: 99.8%  |  Val Acc: 12.3%  (random chance)
Epoch 1000:  Train Acc: 100%   |  Val Acc: 13.1%  (still random)
Epoch 5000:  Train Acc: 100%   |  Val Acc: 14.7%  (barely moving)
Epoch 10000: Train Acc: 100%   |  Val Acc: 15.2%  (patience wearing thin...)
Epoch 15000: Train Acc: 100%   |  Val Acc: 52.4%  (something is happening!)
Epoch 16000: Train Acc: 100%   |  Val Acc: 97.6%  (grokking!)
Epoch 17000: Train Acc: 100%   |  Val Acc: 99.9%  (full generalization)
```

**Phase 2 -- Generalization (slow)**: While the training loss remains flat, the model is silently reorganizing its internal representations. Memorization-based circuits are gradually replaced by structured, algorithm-like circuits that capture the true underlying rule. When these structured representations become strong enough to dominate, validation accuracy jumps abruptly.

### Competing Circuits: Memorization vs. Generalization

A useful framework for understanding grokking is the competition between two internal circuits:

```
Memorization Circuit:
  - Develops quickly (low training loss)
  - Uses many large, specialized weights
  - Acts like a lookup table: input -> memorized output
  - Does not generalize to unseen inputs
  - High weight norm (costly under weight decay)

Generalization Circuit:
  - Develops slowly (requires discovering structure)
  - Uses fewer, structured weights
  - Implements an algorithm: input -> computed output
  - Generalizes to all valid inputs
  - Low weight norm (favored by weight decay)
```

Both circuits develop simultaneously during training, but the memorization circuit dominates initially because it is easier to find via gradient descent. The generalization circuit is the "slower but better" solution that eventually wins once regularization pressure has sufficiently eroded the memorization circuit.

### What Drives the Transition

Several mechanisms have been identified as contributing to grokking:

**Weight Decay as a Compression Force**: Weight decay (L2 regularization) penalizes large weights. The memorization solution requires many large, specialized weights to store each example independently. Over time, weight decay erodes these memorized weights, creating pressure toward simpler, more compressed representations that happen to be the generalizable ones. Without weight decay, grokking often does not occur at all -- the model stays stuck in the memorization phase indefinitely.

**Representation Learning in the Background**: Nanda et al. (2023) performed mechanistic interpretability on grokked networks and found that the model learns interpretable algorithms. For modular addition, the model discovers discrete Fourier transform-like representations -- it learns to map inputs onto circular representations where addition corresponds to rotation. This structured circuit develops slowly alongside the memorization circuit, eventually overtaking it.

**Critical Dataset Size**: Grokking is most pronounced when the training set is small relative to the total number of possible examples. With very large training sets, memorization and generalization happen simultaneously (as in normal training). With very small training sets, the model may never grok because there is insufficient signal to discover the underlying structure.

## Why It Matters

1. **Challenges early stopping**: The standard practice of stopping training when validation loss plateaus would prevent grokking entirely. This raises the question: how many models have been abandoned just before they would have generalized?
2. **Reveals hidden learning dynamics**: Grokking demonstrates that loss curves do not tell the whole story. Significant representational restructuring can occur while observable metrics remain flat.
3. **Informs training duration decisions**: For algorithmic and structured tasks, training far longer than seemingly necessary may yield qualitatively different (and better) solutions.
4. **Connects to mechanistic interpretability**: Grokked networks develop clean, interpretable internal algorithms, making them valuable test cases for understanding what neural networks actually learn.
5. **Illuminates the memorization-generalization spectrum**: Grokking shows that memorization and generalization are not mutually exclusive endpoints but phases in a continuum that a model can traverse with sufficient training.

## Key Technical Details

- Grokking was first observed on modular arithmetic tasks (e.g., (a * b) mod p for prime p) with small training sets (30-50% of all possible input pairs).
- Weight decay is critical: experiments without regularization show memorization without subsequent generalization. Stronger weight decay accelerates grokking.
- The delay between memorization and generalization can span 1-2 orders of magnitude in training steps (e.g., memorization at epoch 100, grokking at epoch 10,000).
- Grokking has been observed beyond toy tasks -- in small transformers, MLPs, and on tasks including permutation groups, polynomial regression, and sparse parity.
- The "slingshot mechanism" (Thilak et al., 2022) suggests that periodic spikes in loss during the plateau phase drive the transition by destabilizing memorized representations.
- Larger models tend to grok faster (fewer steps between memorization and generalization), consistent with the idea that excess capacity makes it easier to develop structured representations.
- Learning rate also affects grokking: too high prevents stable memorization, too low prevents the representational restructuring needed for generalization.
- Weight norm is a useful progress measure during the plateau: even when validation accuracy is flat, steadily decreasing weight norm indicates the generalization circuit is gaining strength relative to the memorization circuit.
- Data augmentation can accelerate grokking by providing additional signal about the underlying structure while keeping the dataset small enough to trigger the memorization-first dynamic.
- Grokking has been reproduced across different optimizers (SGD, Adam, AdamW), confirming that it is a property of the learning dynamics rather than an artifact of a specific optimizer.
- The phenomenon connects to the broader concept of phase transitions in learning: abrupt qualitative changes in model behavior that emerge from gradual quantitative changes in parameters.

## Common Misconceptions

- **"Grokking means the model wasn't learning during the plateau."** The model is actively reorganizing its internal representations throughout the plateau. The weight norm typically decreases steadily during this phase, indicating that structured representations are replacing memorized ones. The flat validation curve conceals significant internal computation.

- **"Grokking only happens on toy problems."** While the most dramatic examples come from algorithmic tasks, grokking-like dynamics (delayed generalization after apparent convergence) have been observed in larger-scale settings, including language model training on structured reasoning tasks. The phenomenon may be more common than recognized but harder to detect at scale.

- **"You should always train longer in case grokking happens."** Grokking requires specific conditions: sufficient regularization, a task with learnable structure, and a training set small enough that memorization is the initial easy solution. For large-scale LLM pre-training on diverse internet text, standard convergence behavior is expected. Grokking is most relevant for fine-tuning on small, structured datasets.

- **"Grokking is just delayed convergence."** It is qualitatively different. In delayed convergence, the model gradually improves. In grokking, the model transitions abruptly from chance-level to near-perfect validation accuracy, reflecting a phase transition in the learned representations rather than incremental improvement.

- **"Grokking requires a special architecture."** Grokking has been observed in MLPs, transformers, CNNs, and other architectures. It is a property of the optimization dynamics and task structure, not of any particular network architecture. Any sufficiently parameterized model trained on a structured task with adequate regularization can exhibit grokking.

## Connections to Other Concepts

- **Regularization (Weight Decay)**: The primary driver of grokking. Without regularization pressure to compress representations, models remain stuck in memorization.
- **Early Stopping**: Grokking directly challenges early stopping heuristics, which would terminate training during the memorization plateau.
- **Mechanistic Interpretability**: Grokked networks provide clean examples of interpretable algorithms emerging in neural networks, making them ideal subjects for circuit-level analysis.
- **Scaling Laws**: Larger models grok faster, connecting to broader observations about how model capacity affects learning dynamics.
- **Catastrophic Forgetting**: Both phenomena involve dramatic shifts in learned representations, though in opposite directions -- grokking builds structure, forgetting destroys it.
- **Curriculum Learning**: Both concern the dynamics of when and how models learn. Curriculum learning manipulates data ordering to control learning dynamics, while grokking reveals that the model's internal curriculum can be dramatically different from the external loss curve.
- **Double Descent**: Another counter-intuitive training phenomenon where test loss first decreases, then increases (overfitting), then decreases again. Grokking and double descent may share underlying mechanisms related to phase transitions in representation structure.

## Diagrams and Visualizations

*See the canonical grokking training curves in: [Power et al., "Grokking: Generalization Beyond Overfitting on Small Algorithmic Datasets" (arXiv:2201.02177)](https://arxiv.org/abs/2201.02177), Figure 1, which shows the dramatic phase transition where training accuracy reaches 100% early while validation accuracy remains at chance for thousands of steps before suddenly jumping to near-perfect.*

*See the mechanistic interpretability analysis of grokked networks in: [Nanda et al., "Progress Measures for Grokking via Mechanistic Interpretability" (arXiv:2301.05217)](https://arxiv.org/abs/2301.05217) -- includes visualizations of the Fourier-based circular representations that the network learns for modular arithmetic, showing the transition from memorization circuits to structured generalization circuits.*

*See also the grokking phase diagram at: [Liu et al., "Omnigrok: Grokking Beyond Algorithmic Data" (arXiv:2210.01117)](https://arxiv.org/abs/2210.01117) -- includes figures mapping the relationship between weight decay strength, dataset size fraction, and the onset of grokking.*

## Further Reading

- Power et al., "Grokking: Generalization Beyond Overfitting on Small Algorithmic Datasets" (2022) -- the original paper documenting grokking
- Nanda et al., "Progress Measures for Grokking via Mechanistic Interpretability" (2023) -- reveals the Fourier-based algorithms learned by grokked networks
- Thilak et al., "The Slingshot Mechanism" (2022) -- identifies periodic loss spikes as catalysts for the grokking transition
- Liu et al., "Omnigrok: Grokking Beyond Algorithmic Data" (2022) -- extends grokking observations to non-algorithmic tasks
- Varma et al., "Explaining Grokking Through Circuit Efficiency" (2023) -- analyzes grokking through the lens of competing memorization vs. generalization circuits
- Davies et al., "Unifying Grokking and Double Descent" (2023) -- connects grokking to other surprising generalization phenomena in deep learning
