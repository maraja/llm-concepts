# Cross-Entropy Loss

**One-Line Summary**: Cross-entropy loss is the objective function that drives LLM training by measuring how "surprised" the model is by the actual next token, rooted in information theory's concept of encoding efficiency.

**Prerequisites**: Basic probability (probability distributions, conditional probability), logarithms, the concept of a loss function in machine learning, how LLMs produce probability distributions over vocabularies.

## What Is Cross-Entropy Loss?

Imagine you are a weather forecaster. Every day, you announce the probability of rain. If you say "90% chance of rain" and it rains, your prediction was good -- you should not be penalized much. But if you say "5% chance of rain" and it rains, your prediction was terrible -- you should be penalized heavily.

![Graph of the negative log-likelihood function showing how cross-entropy heavily penalizes low-probability predictions for the correct class (steep curve near zero) and lightly penalizes high-probability correct predictions](https://upload.wikimedia.org/wikipedia/commons/thumb/c/c6/Cross_entropy_-_loss_function.svg/1200px-Cross_entropy_-_loss_function.svg.png)
*Source: [Wikimedia Commons -- Cross-Entropy Loss Function](https://commons.wikimedia.org/wiki/File:Cross_entropy_-_loss_function.svg)*


Cross-entropy loss works exactly this way. After the model reads a sequence of tokens and predicts a probability distribution over the entire vocabulary for the next token, cross-entropy measures **how much probability the model assigned to the token that actually appeared**. If the model assigned high probability to the correct token, the loss is low. If the model assigned almost zero probability to the correct token, the loss is enormous.

This simple idea -- reward confident correct predictions, heavily penalize confident wrong predictions -- is the engine that drives all of LLM pre-training.

## How It Works


![Diagram of the softmax output layer showing logits converted to probabilities, with cross-entropy loss computed against the one-hot target distribution](https://jalammar.github.io/images/t/output_target_probability_distributions.png)
*Source: [Jay Alammar -- The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)*

### The Mathematical Formula

For a single token prediction, the cross-entropy loss is:

$$\mathcal{L} = -\log P_\theta(x_t | x_{<t})$$

where $P_\theta(x_t | x_{<t})$ is the probability the model (with parameters $\theta$) assigns to the correct token $x_t$ given all preceding tokens $x_{<t}$.

For a full sequence of $T$ tokens, the loss is averaged:

$$\mathcal{L} = -\frac{1}{T} \sum_{t=1}^{T} \log P_\theta(x_t | x_{<t})$$

More generally, cross-entropy between two probability distributions $p$ (the true distribution) and $q$ (the model's predicted distribution) is defined as:

$$H(p, q) = -\sum_{i} p(i) \log q(i)$$

In next-token prediction, the "true distribution" $p$ is a one-hot vector (all mass on the actual next token), which simplifies the formula to the negative log-probability of the correct token.

### Step-by-Step Breakdown

1. **The model produces logits**: For each position, the transformer outputs a vector of raw scores (logits) $z \in \mathbb{R}^{|V|}$, one per vocabulary token.
2. **Softmax converts logits to probabilities**: $P(x_i) = \frac{e^{z_i}}{\sum_j e^{z_j}}$. This ensures all values are between 0 and 1 and sum to 1.
3. **Extract the probability of the correct token**: Look up $P(x_t)$, the probability assigned to the actual next token.
4. **Take the negative logarithm**: $\mathcal{L} = -\log P(x_t)$.
5. **Average across all positions**: Sum the losses for all token positions in the batch and divide by the total number of tokens.

### The Negative Log-Likelihood Interpretation

Cross-entropy loss is equivalent to **negative log-likelihood (NLL)**. Maximizing the likelihood of the data (making the model's probability distribution match the data as closely as possible) is the same as minimizing the negative log-likelihood, which is the same as minimizing cross-entropy. These three formulations are mathematically identical:

$$\text{Minimize } \mathcal{L} = \text{Minimize NLL} = \text{Maximize Likelihood}$$

### Connection to Information Theory

Claude Shannon's information theory provides the deepest intuition. The **information content** (or "surprisal") of an event with probability $p$ is $-\log_2 p$ bits. A certain event ($p=1$) carries 0 bits of information -- no surprise. A rare event ($p=0.001$) carries about 10 bits -- enormous surprise.

Cross-entropy measures the **average number of bits needed to encode the true data if you use the model's predicted distribution** for your encoding scheme. A perfect model would achieve the **entropy** $H(p)$ of the true distribution -- the theoretical minimum. The gap between cross-entropy and entropy is the **KL divergence**:

$$H(p, q) = H(p) + D_{KL}(p \| q)$$

Training minimizes this gap, pushing the model's distribution closer to the true data distribution.

### Connection to Perplexity

Perplexity, the most common evaluation metric for language models, is simply the exponentiation of cross-entropy:

$$\text{Perplexity} = e^{\mathcal{L}} = e^{-\frac{1}{T}\sum_t \log P(x_t | x_{<t})}$$

If using base-2 logarithms, perplexity equals $2^{H(p,q)}$. Perplexity can be interpreted as "the effective number of tokens the model is choosing between at each step." A perplexity of 10 means the model is, on average, as uncertain as if it were choosing uniformly among 10 options. Lower perplexity means better modeling.

### How Cross-Entropy Drives Learning

The gradient of cross-entropy loss with respect to the logits has an elegant form. For the softmax output $q_i = P(x_i)$ and the true one-hot label $p_i$:

*See also the relationship between cross-entropy, KL divergence, and entropy visualized at: [Chris Olah -- Visual Information Theory](https://colah.github.io/posts/2015-09-Visual-Information/) -- provides intuitive visual explanations of information-theoretic concepts underlying cross-entropy loss.*


$$\frac{\partial \mathcal{L}}{\partial z_i} = q_i - p_i$$

This means:
- For the **correct token**: the gradient is $q_t - 1$ (negative, pushing logits up, increasing probability).
- For **incorrect tokens**: the gradient is $q_i - 0 = q_i$ (positive, pushing logits down, decreasing probability).

The model learns to increase the probability of correct tokens and decrease the probability of incorrect tokens, with the magnitude of adjustment proportional to how wrong the current prediction is. This is remarkably clean and numerically stable.

## Why It Matters

Cross-entropy loss is the reason LLMs learn. It is the single number that the entire training process seeks to minimize. Every architectural innovation, every data curation decision, every hyperparameter choice -- all of these matter only insofar as they help reduce cross-entropy on the training data (while generalizing to unseen data).

The choice of cross-entropy is not arbitrary. It is the theoretically optimal loss function for classification problems (including next-token prediction) under the framework of maximum likelihood estimation. Using a different loss function, such as mean squared error on probability values, would produce worse gradients, slower convergence, and inferior models.

## Key Technical Details

- **Numerical stability**: In practice, the log-softmax is computed in a single numerically stable operation (LogSoftmax) rather than computing softmax and then taking the log separately. This avoids underflow and overflow issues.
- **Label smoothing**: Sometimes a small amount of probability (e.g., 0.1) is spread across all tokens to prevent the model from becoming overconfident. This modifies the one-hot target distribution.
- **Vocabulary size impact**: With a vocabulary of 50,000-128,000 tokens, the initial loss at random initialization is approximately $\log(|V|)$. For $|V| = 100,000$, this is about 11.5 nats.
- **Reduction to entropy**: As training progresses, cross-entropy approaches the entropy of natural language, which is estimated at roughly 1.0-1.5 bits per character (or equivalently, roughly 3-7 nats per token depending on tokenization).
- **Loss masking**: Padding tokens and sometimes special tokens are masked out of the loss computation to avoid training on meaningless signals.

## Common Misconceptions

- **"Cross-entropy measures accuracy."** It does not measure whether the top prediction is correct (that would be accuracy). It measures the probability mass assigned to the correct token. A model can have low accuracy but decent cross-entropy if it spreads probability reasonably.
- **"Lower loss always means a better model."** On training data, lower loss could indicate overfitting. What matters is loss on held-out validation data.
- **"Cross-entropy is specific to language models."** It is the standard loss function for virtually all classification tasks in deep learning, from image classification to speech recognition.
- **"The loss should reach zero."** Natural language has inherent entropy -- genuine uncertainty about the next word. A loss of zero would mean the model perfectly predicts every token, which is impossible for natural text and would indicate catastrophic overfitting.

## Connections to Other Concepts

- **Pre-Training**: Cross-entropy loss is the objective that the entire pre-training process minimizes.
- **Perplexity**: The direct exponential of cross-entropy; the standard evaluation metric for language models.
- **Backpropagation**: The mechanism by which gradients of cross-entropy flow backward through the network to update parameters.
- **Softmax Temperature**: Scaling logits before softmax changes the "sharpness" of the distribution, directly affecting cross-entropy.
- **Knowledge Distillation**: Uses cross-entropy between a student model's distribution and a teacher model's distribution (soft targets) rather than one-hot labels.
- **KL Divergence**: Used in RLHF to keep the fine-tuned model close to the pre-trained model; intimately related to cross-entropy.

## Further Reading

- Shannon, C.E. (1948). "A Mathematical Theory of Communication" -- The foundational paper that defined entropy and cross-entropy, establishing the information-theoretic basis for everything in this field.
- Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning*, Chapter 3.13 -- Provides a rigorous treatment of cross-entropy in the context of maximum likelihood estimation for deep learning.
- Meister, C., & Cotterell, R. (2021). "Language Model Evaluation Beyond Perplexity" -- Explores the nuances and limitations of cross-entropy/perplexity as evaluation metrics for language models.
