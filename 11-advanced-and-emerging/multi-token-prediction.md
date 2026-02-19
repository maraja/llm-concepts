# Multi-Token Prediction

**One-Line Summary**: Multi-token prediction trains language models to predict several future tokens simultaneously from each position, producing richer internal representations and enabling faster inference through speculative self-decoding.

**Prerequisites**: Transformer architecture, next-token prediction, autoregressive generation, training objectives, speculative decoding

## What Is Multi-Token Prediction?

Imagine learning to play chess by only ever thinking one move ahead. You might learn local tactics -- forks, pins, skewers -- but you would struggle with strategy that requires envisioning positions three or four moves into the future. Now imagine training yourself to always visualize the next four moves as a sequence. You would develop a qualitatively different kind of understanding: one that incorporates planning, anticipation, and longer-range coherence. That is the intuition behind multi-token prediction.

Standard language model training uses a next-token prediction (NTP) objective: at each position in the sequence, the model predicts only the immediately following token. Multi-token prediction (MTP) extends this by requiring the model to simultaneously predict tokens at positions t+1, t+2, t+3, and t+4 (or more) from the same hidden representation at position t. Each future position gets its own lightweight prediction head, but all heads share the same transformer backbone.

This deceptively simple change has profound implications. By forcing the model to anticipate further into the future, MTP encourages the backbone to build representations that encode not just "what comes next" but "what trajectory the text is on." Meta's Gloeckle et al. (2024) demonstrated these benefits at scale, and DeepSeek-V3 adopted MTP as a key architectural innovation in one of the most capable open-weight models.

## How It Works

### Architecture and Training

The MTP architecture augments a standard transformer with multiple prediction heads. Here is the conceptual structure:

```
Input tokens:     [The] [cat] [sat] [on] [the] [mat]

Transformer backbone processes all positions
                          |
              Shared hidden states h_t
                    /    |    \     \
                 Head1  Head2  Head3  Head4
                  |      |      |      |
               pred(t+1) pred(t+2) pred(t+3) pred(t+4)
```

Each prediction head is a small neural network (typically one or two transformer layers plus a linear projection to the vocabulary) that takes the backbone's hidden state and predicts a specific future token. The training loss is the sum (or weighted sum) of cross-entropy losses across all heads:

```
L_total = L_{t+1} + L_{t+2} + L_{t+3} + L_{t+4}
```

In practice, the primary head (t+1) still receives the most gradient signal and carries the most weight, while the auxiliary heads (t+2, t+3, t+4) provide supplementary training signal. The key insight is that gradients from the auxiliary heads flow back through the shared backbone, enriching its representations.

### Why Predicting Further Ahead Helps

Consider the sentence: "The capital of France is ___." Predicting the next token ("Paris") is relatively easy -- it is a high-frequency factual association. But predicting t+2 (","), t+3 ("which"), t+4 ("is") requires understanding that a relative clause or additional context is likely to follow. The model must develop representations that encode not just the answer but the discourse structure.

This effect is especially pronounced in code generation:

```python
# Given context: def fibonacci(n):
# Predicting t+1: "if" (common function start)
# Predicting t+2: "n" (the parameter)
# Predicting t+3: "<=" (comparison)
# Predicting t+4: "1" (base case boundary)
```

To predict all four tokens correctly, the model must internally represent that a recursive function with a base case is the most likely continuation. This is a form of implicit planning that NTP alone does not directly incentivize.

### Speculative Self-Decoding at Inference

A major practical benefit of MTP is that the auxiliary prediction heads can be used for speculative decoding at inference time -- without needing a separate draft model. The process works as follows:

1. Run the model forward, producing predictions from all four heads.
2. The primary head's prediction (t+1) is accepted.
3. The auxiliary heads' predictions (t+2, t+3, t+4) are treated as speculative drafts.
4. Verify the drafts by running a single forward pass over the drafted sequence.
5. Accept all consecutive correct predictions; reject from the first error onward.

This achieves a 1.5-2x inference speedup on average because multiple tokens are generated per forward pass. Unlike external speculative decoding (which requires a separate smaller model), MTP's self-speculation comes "for free" from the training process.

### Training Implementation Details

In practice, MTP training requires careful handling of the loss computation across heads:

```python
# Simplified MTP training loop
def compute_mtp_loss(model, input_ids):
    hidden_states = model.backbone(input_ids)  # Shared forward pass

    total_loss = 0
    for k, head in enumerate(model.prediction_heads):
        # Head k predicts token at position t+k+1
        logits_k = head(hidden_states)
        # Shift targets by k+1 positions
        targets_k = input_ids[:, k+1:]
        logits_k = logits_k[:, :targets_k.size(1), :]
        loss_k = cross_entropy(logits_k, targets_k)
        total_loss += loss_k  # Equal weighting (or use decay: loss_k * 0.8^k)

    return total_loss
```

The auxiliary heads can use equal weighting or a decay schedule (where farther-ahead predictions receive less weight). DeepSeek-V3 found that equal weighting works well in practice, though the optimal scheme may be task-dependent.

## Why It Matters

1. **Richer representations**: The backbone develops hidden states that encode multi-step futures, leading to better performance on tasks requiring planning, coherence, and complex reasoning.
2. **Significant inference speedup**: Speculative self-decoding provides 1.5-2x faster generation without the engineering complexity of maintaining a separate draft model.
3. **Modest training overhead**: The auxiliary heads add only ~10-15% to training compute, while the backbone (which dominates cost) remains unchanged in size. This is an excellent cost-benefit ratio.
4. **Proven at scale**: DeepSeek-V3's adoption of MTP in a state-of-the-art production model validates the approach beyond research settings.
5. **Especially strong for code**: Code generation benefits disproportionately because code has rigid syntactic structure where planning multiple tokens ahead is both feasible and rewarding.

## Key Technical Details

- The standard configuration uses 4 auxiliary heads (predicting t+1 through t+4), though this is a tunable hyperparameter. Diminishing returns appear beyond 4-6 heads.
- Each auxiliary head is lightweight -- typically 1-2 transformer layers plus a vocabulary projection -- compared to the 40-80+ layer backbone.
- The auxiliary heads share the backbone's embedding and unembedding matrices to avoid parameter explosion.
- During training, the primary NTP loss is not downweighted; auxiliary losses are additive, ensuring that standard next-token performance is not degraded.
- MTP's benefit is strongest in the low-to-mid data regime; with infinite data, NTP can eventually learn similar representations, but MTP gets there faster and with less data.
- Gloeckle et al. showed that MTP-trained models outperform NTP-only models on code benchmarks (HumanEval, MBPP) by 5-15% at equivalent model sizes and compute budgets.
- The speculative decoding acceptance rate depends on text predictability: boilerplate code and formulaic text see higher speedups than creative writing.

## Common Misconceptions

- **"MTP trains a separate model for each future token."** All prediction heads share the same transformer backbone. Only the small head networks are separate. The vast majority of parameters (>95%) are shared, and the critical benefit comes from the enriched backbone representations.

- **"MTP makes training much more expensive."** The additional compute is modest (10-15%), because the auxiliary heads are small relative to the backbone. The backbone forward and backward pass -- which dominates compute -- is identical in cost to standard NTP training. The memory overhead for the auxiliary heads is similarly small.

- **"Predicting further ahead is just harder, not more useful."** While accuracy on t+3 and t+4 is indeed lower than t+1, the gradient signal from attempting these predictions still forces the backbone to build richer representations. Even "failed" predictions at t+4 provide useful training signal about long-range dependencies.

- **"You need MTP-trained models to use speculative decoding."** Standard speculative decoding works with any pair of draft and target models. MTP's advantage is that it provides built-in draft predictions, eliminating the need to train and deploy a separate draft model.

## Connections to Other Concepts

- **Next-Token Prediction**: MTP is a direct generalization of NTP. The t+1 head in MTP is functionally identical to the standard NTP objective.
- **Speculative Decoding**: MTP provides a natural mechanism for self-speculation, complementing external speculative decoding approaches.
- **Training Objectives (BERT-style MLM)**: MTP can be seen as a middle ground between autoregressive NTP and bidirectional objectives -- it maintains autoregressive structure while extracting richer signal per position.
- **DeepSeek-V3 / Mixture of Experts**: DeepSeek-V3 combined MTP with MoE architecture, demonstrating that MTP composes well with other architectural innovations.
- **Knowledge Distillation**: MTP's auxiliary heads resemble multi-task distillation, where the model "teaches itself" about future structure.

## Diagrams and Visualizations

*Recommended visual: Multi-token prediction architecture showing multiple prediction heads predicting n future tokens simultaneously — see [Gloeckle et al. Multi-Token Prediction Paper (arXiv:2404.19737)](https://arxiv.org/abs/2404.19737)*

*Recommended visual: Self-speculative decoding using multi-token heads as draft predictions for verification — see [Meta AI Multi-Token Prediction Blog](https://arxiv.org/abs/2404.19737)*

## Further Reading

- Gloeckle et al., "Better & Faster Large Language Models via Multi-token Prediction" (2024) -- Meta's foundational paper demonstrating MTP at scale
- DeepSeek-AI, "DeepSeek-V3 Technical Report" (2024) -- production-scale adoption of MTP as a core architectural choice
- Qi et al., "Speculative Decoding with Multi-Token Prediction" (2024) -- detailed analysis of MTP's inference speedup characteristics
- Stern et al., "Insertion Transformer" (2019) -- earlier work on predicting multiple tokens, providing historical context for MTP
