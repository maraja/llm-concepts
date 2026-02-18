# Activation Functions in LLMs

**One-Line Summary**: Activation functions introduce non-linearity into neural networks, enabling them to learn complex patterns, and the evolution from ReLU to GELU to SwiGLU represents a progression toward smoother, gated functions that improve large language model training dynamics and performance.

**Prerequisites**: Understanding of neural network basics (linear transformations, gradient descent), the feed-forward network layer in Transformers, and the concept of a derivative/gradient.

## What Are Activation Functions?

Without activation functions, a neural network is just a stack of linear transformations -- and a stack of linear transformations collapses into a single linear transformation. No matter how many layers you add, the network can only learn linear relationships: $y = Wx + b$.

An activation function $\sigma$ is applied after each linear transformation to introduce **non-linearity**:

$$y = \sigma(Wx + b)$$

This non-linearity is what gives neural networks their extraordinary representational power. With a non-linear activation, even a single hidden layer can theoretically approximate any continuous function (universal approximation theorem). In practice, deeper networks with non-linear activations can learn hierarchical, compositional representations.

Think of it this way: a linear function can draw a straight line through data. An activation function lets the network bend, curve, and fold the space, creating the complex decision boundaries needed to model language.

## How It Works

### ReLU (Rectified Linear Unit)

$$\text{ReLU}(x) = \max(0, x)$$

The simplest modern activation. It passes positive values unchanged and zeros out negative values. Its derivative is 1 for $x > 0$ and 0 for $x < 0$.

**Strengths**: Simple, fast to compute, produces sparse activations (many zeros), and avoids the vanishing gradient problem that plagued sigmoid/tanh for positive inputs.

**Weaknesses**: The "dying ReLU" problem -- if a neuron's inputs consistently produce negative values, its gradient is always zero, and it stops learning permanently. The sharp corner at $x = 0$ can cause optimization difficulties. ReLU was used in some early Transformer work but is rarely used in modern LLMs.

### GELU (Gaussian Error Linear Unit)

$$\text{GELU}(x) = x \cdot \Phi(x) = x \cdot \frac{1}{2}\left[1 + \text{erf}\left(\frac{x}{\sqrt{2}}\right)\right]$$

where $\Phi(x)$ is the cumulative distribution function of the standard Gaussian. A commonly used approximation:

$$\text{GELU}(x) \approx 0.5x\left(1 + \tanh\left[\sqrt{2/\pi}(x + 0.044715x^3)\right]\right)$$

GELU can be understood as a **smooth, stochastic version of ReLU**. Instead of the hard threshold at 0, GELU smoothly interpolates: inputs with large positive values pass through almost unchanged, large negative values are nearly zeroed, and values near zero are partially suppressed.

**Intuition**: GELU can be interpreted as "multiply the input by the probability that it would survive a random dropout based on its magnitude." Larger values are more likely to "survive," creating a soft gating effect.

**Used in**: BERT, GPT-2, GPT-3, and many models from 2018-2022.

### SiLU / Swish

$$\text{SiLU}(x) = x \cdot \sigma(x) = \frac{x}{1 + e^{-x}}$$

where $\sigma(x)$ is the sigmoid function. SiLU (Sigmoid Linear Unit) is also known as Swish, as proposed by Ramachandran et al. (2017) at Google.

SiLU is closely related to GELU -- both are smooth, non-monotonic functions that allow small negative values to pass through (unlike ReLU). SiLU dips slightly below zero for negative inputs before returning to zero, creating a small "bump" that some researchers believe helps with optimization.

**Key property**: SiLU is **non-monotonic** -- it decreases slightly for moderately negative inputs before returning to zero. This means it does not simply suppress all negative values; it has a nuanced response.

**Used in**: PaLM, LLaMA 1 (as part of SwiGLU).

### GLU (Gated Linear Unit) and Its Variants

The **Gated Linear Unit** introduces a multiplicative gating mechanism:

$$\text{GLU}(x) = (W_1 x) \odot \sigma(W_3 x)$$

where $\odot$ is element-wise multiplication, $W_1$ and $W_3$ are separate learned projections, and $\sigma$ is a sigmoid gate. One projection produces the "content" and the other produces the "gate" that controls how much of each content element passes through.

This is powerful because the gating is **input-dependent** and **element-wise**: different neurons are gated by different amounts based on the specific input, giving the network fine-grained control over information flow.

### SwiGLU (The Modern Standard)

$$\text{SwiGLU}(x) = (\text{SiLU}(W_1 x)) \odot (W_3 x)$$

SwiGLU replaces the sigmoid gate in GLU with SiLU applied to one branch, while the other branch is a simple linear projection. The full FFN with SwiGLU:

$$\text{FFN}_{SwiGLU}(x) = W_2 \cdot \left[\text{SiLU}(W_1 x) \odot (W_3 x)\right]$$

**Why three weight matrices**: Standard FFNs have two matrices ($W_1$ up-projection, $W_2$ down-projection). SwiGLU adds a third ($W_3$) for the gate. To maintain the same parameter count, the hidden dimension $d_{ff}$ is reduced from $4 \times d_{model}$ to approximately $\frac{8}{3} \times d_{model}$.

**Why SwiGLU dominates**: Shazeer (2020) systematically compared GLU variants and found that SwiGLU and GeGLU (GELU-based gating) consistently outperformed standard activations (ReLU, GELU) on language modeling tasks. SwiGLU provides better loss at the same parameter count, making it a "free" improvement.

**Used in**: LLaMA 2, LLaMA 3, Mistral, Gemma, PaLM 2, and most modern LLMs.

## Why It Matters

### The Evolution Tells a Story

The progression of activation functions in LLMs reflects deepening understanding of optimization dynamics:

1. **ReLU** (2010s): Simple, effective, but harsh. The hard zero boundary creates dead neurons and sharp gradients.
2. **GELU** (2016): Smooth approximation of ReLU. Better for optimization because gradients are continuous everywhere.
3. **SiLU/Swish** (2017): Discovered via neural architecture search. Non-monotonic, smooth, self-gated.
4. **SwiGLU** (2020): Combines smooth activation with an explicit gating mechanism. The gating allows the network to learn more complex functions within each FFN.

Each step improved training stability and final model quality, enabling the training of larger models. The choice of activation function might seem minor, but at the scale of hundreds of billions of parameters trained on trillions of tokens, small efficiency differences compound into meaningful quality gaps.

### Gating as a Design Principle

The success of GLU variants highlights a broader principle: **multiplicative interactions** (gating) are powerful. Gating lets one computation control another, creating richer function approximation within a single layer. This principle appears throughout modern architectures -- not just in FFN activations but also in LSTM gates, attention mechanisms (which are a form of gating), and mixture-of-experts routers.

## Key Technical Details

- **Smoothness matters**: GELU, SiLU, and SwiGLU are all infinitely differentiable ($C^{\infty}$), unlike ReLU which has a discontinuous derivative at $x = 0$. Smooth gradients help optimizers navigate the loss landscape more effectively.
- **Non-monotonicity**: Both GELU and SiLU are non-monotonic (they dip slightly below zero before returning). This allows the network to suppress certain activations in a nuanced way.
- **Computational cost**: SwiGLU is more expensive per FLOP than ReLU (due to the extra multiplication and the third weight matrix), but the quality improvement per parameter makes it more compute-efficient overall.
- **Sparsity**: ReLU produces exact zeros for negative inputs (structural sparsity). GELU and SiLU produce near-zero but non-zero values. Some researchers have explored combining SwiGLU with explicit sparsification.
- **Activation function is applied in the FFN only**: In the Transformer, the attention mechanism does not use an activation function (softmax in attention serves a different role). The activation function appears only in the feed-forward network.
- **No single "best" activation**: While SwiGLU is the current standard, the optimal choice can depend on model size, training setup, and task. The field continues to explore alternatives.

## Common Misconceptions

- **"ReLU is still a good default for LLMs."** While ReLU works, it is significantly outperformed by GELU and SwiGLU for language modeling at scale. The dying neuron problem and non-smooth gradients make it suboptimal for deep Transformers.
- **"The activation function is a minor implementation detail."** At scale, the choice of activation function measurably affects training loss and downstream performance. The switch from GELU to SwiGLU in LLaMA 2 was one of several changes that improved quality.
- **"SwiGLU has more parameters, so it's unfairly better."** When compared at equal parameter count (by reducing the hidden dimension to compensate for the third weight matrix), SwiGLU still outperforms standard activations. The improvement comes from the gating mechanism, not the extra parameters.
- **"Activation functions are only about non-linearity."** While non-linearity is the fundamental purpose, modern activation functions also shape gradient flow, control sparsity, and enable gating -- properties that go beyond simple non-linearity.

## Connections to Other Concepts

- **Feed-Forward Networks**: Activation functions are a core component of the FFN sub-layer (see `feed-forward-networks.md`).
- **Logits and Softmax**: Softmax is itself an activation function, used at the output layer rather than in hidden layers (see `logits-and-softmax.md`).
- **Residual Connections**: The output of activated FFN layers is added to the residual stream (see `residual-connections.md`).
- **Layer Normalization**: Normalization and activation functions interact to determine the distribution of values flowing through the network (see `layer-normalization.md`).
- **Mixture of Experts**: Each expert in an MoE model contains an FFN with its own activation function (see `mixture-of-experts.md`).

## Further Reading

- "Gaussian Error Linear Units (GELUs)" -- Hendrycks and Gimpel, 2016 (the original GELU paper)
- "Searching for Activation Functions" -- Ramachandran, Zoph, and Le, 2017 (discovers Swish/SiLU via architecture search)
- "GLU Variants Improve Transformer" -- Noam Shazeer, 2020 (systematic comparison leading to SwiGLU adoption)
