# Learning Rate Scheduling

**One-Line Summary**: Learning rate scheduling -- gradually warming up, then systematically decaying the learning rate during training -- is a critical technique that prevents early training instability and ensures the model converges to a good minimum rather than oscillating around one.

**Prerequisites**: Gradient descent and the role of learning rate, optimizer basics (Adam/AdamW), the concept of loss landscape and convergence, basic understanding of pre-training dynamics.

## What Is Learning Rate Scheduling?

Imagine you are searching for the lowest point in a vast, foggy valley. If you take enormous strides, you will quickly reach the general area of the valley floor but then keep overshooting it, bouncing back and forth across the bottom. If you take tiny steps from the start, you will make precise progress but it will take an eternity to get anywhere.

![Comparison of learning rate schedules: constant, step decay, exponential decay, cosine annealing, and warmup + cosine decay, showing how each schedule modulates the learning rate over training steps](https://www.researchgate.net/publication/338427616/figure/fig3/AS:845546676903938@1578597896804/Different-learning-rate-schedules.png)
*Source: [ResearchGate -- Different Learning Rate Schedules](https://www.researchgate.net/)*


The optimal strategy is to start with medium strides to get your bearings, build up to large strides once you know the general direction, and then gradually shrink your steps as you approach the minimum so you can settle precisely into it. This is exactly what learning rate scheduling does.

A **constant learning rate** is almost never optimal for LLM training. Instead, the learning rate follows a carefully designed **schedule** that changes throughout training, typically involving three phases: warmup, stable (or peak), and decay.

## How It Works


*See the cosine annealing with warm restarts diagram in: [Loshchilov & Hutter, "SGDR: Stochastic Gradient Descent with Warm Restarts" (arXiv:1608.03983)](https://arxiv.org/abs/1608.03983), Figure 1, which shows the cyclic cosine decay pattern that forms the basis of modern LLM learning rate schedules.*

### Why Constant Learning Rate Fails

At the beginning of training, model parameters are randomly initialized. The gradients in this regime are large, noisy, and poorly calibrated -- the loss landscape looks very different from what it will look like later. A high constant learning rate causes destructive, oversized updates that can destabilize training (loss spikes, divergence). A low constant learning rate avoids instability but makes training painfully slow and may get stuck in suboptimal regions.

Even after the early instability passes, a constant learning rate prevents fine convergence. The model keeps making updates of the same magnitude, oscillating around a minimum rather than settling into it.

### Phase 1: Warmup

During warmup, the learning rate increases linearly from near-zero to the peak learning rate over a set number of steps:

$$\eta_t = \eta_{\text{peak}} \cdot \frac{t}{T_{\text{warmup}}}$$

where $t$ is the current step and $T_{\text{warmup}}$ is the total number of warmup steps.

**Why warmup is necessary:**

1. **Adam's moment estimates are unreliable initially.** The first and second moments ($m$ and $v$) are initialized to zero and need hundreds to thousands of steps to become accurate estimates. Large learning rates combined with inaccurate moment estimates produce erratic updates.
2. **Early gradients are noisy and large.** The randomly initialized model produces essentially random predictions, leading to large gradients. Multiplying these by a high learning rate causes destructive parameter updates.
3. **Stabilizes training across distributed setups.** With thousands of GPUs, batch statistics and gradient aggregation need time to stabilize. Warmup provides this grace period.

Typical warmup durations: 500 to 2,000 steps for large models (sometimes up to 1-2% of total training steps).

### Phase 2: Cosine Decay

The most common decay schedule for LLM training is **cosine annealing**, where the learning rate follows a cosine curve from the peak down to a minimum value:

$$\eta_t = \eta_{\text{min}} + \frac{1}{2}(\eta_{\text{peak}} - \eta_{\text{min}}) \left(1 + \cos\left(\frac{t - T_{\text{warmup}}}{T_{\text{total}} - T_{\text{warmup}}} \cdot \pi\right)\right)$$

This produces a smooth, continuous decay that:
- Decreases slowly at first (allowing the model to continue making meaningful progress at the peak rate).
- Accelerates in the middle.
- Slows down again near the end (gentle final convergence).

The minimum learning rate $\eta_{\text{min}}$ is typically set to 10% of the peak rate (e.g., if peak is 3e-4, minimum is 3e-5) or sometimes even lower.

**Why cosine decay works well:**

The cosine shape is not arbitrary. It spends more total training time at higher learning rates (where the model makes the most progress) compared to linear decay, while still approaching a small final learning rate. Empirically, cosine schedules consistently produce better final loss than linear or step decay schedules for transformer training.

### Alternative: Warmup-Stable-Decay (WSD)

An increasingly popular alternative, sometimes called the "trapezoidal" schedule:

1. **Warmup**: Linear increase to peak learning rate (same as above).
2. **Stable**: Maintain the peak learning rate for the majority of training.
3. **Decay**: Rapid decay (linear, cosine, or exponential) in the final 10-20% of training.

$$\eta_t = \begin{cases} \eta_{\text{peak}} \cdot \frac{t}{T_{\text{warmup}}} & \text{if } t < T_{\text{warmup}} \\ \eta_{\text{peak}} & \text{if } T_{\text{warmup}} \leq t < T_{\text{stable\_end}} \\ \eta_{\text{peak}} \cdot f_{\text{decay}}(t) & \text{if } t \geq T_{\text{stable\_end}} \end{cases}$$

This schedule has a practical advantage: the stable phase produces checkpoints that are roughly equivalent to each other, making it easier to decide when to stop training or to branch off for fine-tuning at any point during the stable phase. With cosine decay, each checkpoint is at a different point in the schedule, making comparisons and branching decisions more complex.

### Typical Values for LLM Training

| Parameter | Typical Range | Notes |
|-----------|--------------|-------|
| Peak learning rate | 1e-4 to 6e-4 | Inversely related to model size |
| Warmup steps | 500 to 2,000 | Sometimes expressed as fraction of total steps |
| Total training steps | 100K to 1M+ | Depends on data size and compute budget |
| Minimum learning rate | 1/10 to 1/100 of peak | Too low can waste the final training steps |
| Schedule shape | Cosine or WSD | Cosine is most common; WSD gaining popularity |

### Interaction with Model Size

Larger models generally require **smaller peak learning rates**. This relationship is well-characterized by scaling laws:

*See also the LLM training learning rate schedule from: [Hoffmann et al., "Training Compute-Optimal Large Language Models" (Chinchilla, arXiv:2203.15556)](https://arxiv.org/abs/2203.15556) -- includes the warmup + cosine decay schedule used for Chinchilla training with specific hyperparameter values.*


- GPT-3 175B: peak LR ~ 0.6e-4
- LLaMA 65B: peak LR ~ 1.5e-4
- LLaMA 7B: peak LR ~ 3.0e-4

The intuition is that larger models have more parameters interacting in complex ways, so each individual parameter needs smaller updates to avoid destabilizing the whole system.

## Why It Matters

Learning rate scheduling is one of the few hyperparameter choices that can make the difference between a successful training run and a complete failure. An improperly scheduled learning rate can:

- **Cause training divergence**: Too high early on, the model produces NaN losses and the run is dead.
- **Waste compute**: Too low throughout, the model trains slowly and never reaches its potential in the allocated compute budget.
- **Produce suboptimal models**: Wrong decay shape or duration can leave the model in a worse minimum, reducing the quality of all downstream fine-tuning and deployment.

Given that a single training run can cost tens of millions of dollars, getting the learning rate schedule right is enormously consequential. Teams typically run smaller-scale experiments to validate their schedule before committing to the full run.

## Key Technical Details

- **The learning rate schedule multiplies with Adam's adaptive rate**: The scheduled $\eta_t$ is the base learning rate that Adam then adapts per-parameter. The effective learning rate for a specific parameter is $\eta_t \cdot \frac{\hat{m}}{\sqrt{\hat{v}} + \epsilon}$.
- **Warmup duration is not very sensitive**: Whether you use 500 or 2,000 warmup steps usually does not dramatically change the final model. However, using zero warmup often causes training to fail.
- **Restarting from a checkpoint**: If training is interrupted and resumed, the learning rate schedule must be resumed from the correct step, not restarted. Restarting warmup on a partially trained model can be destructive.
- **Learning rate finder**: Some practitioners use a diagnostic run where the learning rate is gradually increased to find the point where loss starts increasing (the maximum stable learning rate), then set the peak to some fraction of this value.
- **Batch size and learning rate are coupled**: The linear scaling rule suggests that when doubling the batch size, the learning rate should also be doubled (or more precisely, follow a square-root scaling). This relationship is important for distributed training where effective batch sizes can be very large.

## Common Misconceptions

- **"Warmup is just for avoiding NaN losses."** While preventing divergence is one benefit, warmup also helps the optimizer build accurate moment estimates and allows the model to develop meaningful gradient signals before large updates begin.
- **"The exact schedule shape matters a lot."** In practice, the most important factors are the peak learning rate, the warmup duration, and the fact that *some* decay happens. The precise shape (cosine vs. linear vs. polynomial) has a real but secondary effect.
- **"Lower learning rate is always safer."** Too low a learning rate can be worse than too high, because the model under-trains within the compute budget. You waste expensive compute without reaching the model's potential.
- **"You should always decay to zero."** Decaying to exactly zero wastes the final portion of training, since the model is effectively not learning. Decaying to a small but nonzero minimum (e.g., 10% of peak) is standard practice.

## Connections to Other Concepts

- **Adam/AdamW Optimizer**: The learning rate schedule modulates the base rate that Adam adapts per-parameter.
- **Pre-Training**: Learning rate scheduling is a critical hyperparameter of the pre-training process.
- **Scaling Laws**: The optimal peak learning rate scales predictably with model size.
- **Training Stability (Gradient Clipping)**: Learning rate scheduling works alongside gradient clipping to maintain stable training.
- **Mixed Precision Training**: Some precision formats (FP16 especially) are more sensitive to learning rate choices due to limited dynamic range.
- **Fine-Tuning**: Fine-tuning uses a much smaller learning rate than pre-training (often 10-100x smaller), sometimes with its own warmup schedule.

## Further Reading

- Loshchilov, I. & Hutter, F. (2017). "SGDR: Stochastic Gradient Descent with Warm Restarts" -- Introduced cosine annealing with restarts, the foundation for modern cosine decay schedules.
- Goyal, P., et al. (2017). "Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour" -- Established the linear scaling rule for learning rate with batch size and the importance of warmup for large-batch training.
- Hoffmann, J., et al. (2022). "Training Compute-Optimal Large Language Models" (Chinchilla) -- Provides detailed learning rate schedule configurations for large-scale LLM training and demonstrates how schedule choices interact with compute-optimal training decisions.
