# Context Window

**One-Line Summary**: The context window is the fixed-length span of tokens a transformer model can attend to in a single forward pass -- the model's "working memory" that determines how much text it can consider at once.

**Prerequisites**: Understanding of tokenization (how text is converted to tokens), positional encoding (how the model tracks token order), self-attention (how tokens attend to each other), and basic awareness of computational complexity (Big-O notation).

## What Is Context Window?

Think of reading a book through a small window cut in a piece of cardboard. You can slide the window over the page and see whatever text is currently visible, but you cannot see the whole page at once. The size of that window determines how much context you have for understanding any given word.

A transformer's context window works similarly. When you send a prompt to GPT-4, the model can only "see" a certain number of tokens at once. If your conversation plus the model's response exceeds that limit, older content must be dropped or summarized. The context window is the hard upper bound on how much information the model can jointly reason about in a single computation.

This is not merely a technical limitation -- it fundamentally shapes what LLMs can and cannot do. A model with a 4K context window literally cannot read a 10,000-word document in one pass. A model with 128K context can, but may not use it all equally well.

## How It Works

### What Determines Context Length

Three factors constrain the context window:

**1. Positional Encoding**: The model needs a way to represent every position in the sequence. Learned positional embeddings (GPT-2) have a fixed maximum length baked into the architecture. RoPE-based models are more flexible but still degrade beyond training length without extension techniques.

**2. Memory and Compute**: Self-attention has $O(n^2)$ complexity in both time and memory with respect to sequence length $n$. For a sequence of length $n$:

$$\text{Attention Memory} \propto n^2 \cdot d_{\text{head}} \cdot h$$

where $d_{\text{head}}$ is the head dimension and $h$ is the number of heads. Doubling the context length quadruples the attention computation. For a model with 32 attention layers, 32 heads, and head dimension 128:
- At 4K tokens: ~2 GB of attention memory
- At 128K tokens: ~2 TB of attention memory

This is why efficient attention variants matter enormously.

**3. Training Data**: A model must be trained on (or fine-tuned with) sequences of the target context length. Attention patterns for long-range dependencies can only be learned if long-range dependencies are present in the training data.

### The Evolution of Context Length

| Year | Model | Context Length |
|------|-------|---------------|
| 2018 | BERT | 512 tokens |
| 2019 | GPT-2 | 1,024 tokens |
| 2020 | GPT-3 | 2,048 tokens |
| 2023 | GPT-4 | 8,192 / 32K tokens |
| 2023 | Claude 2 | 100K tokens |
| 2024 | GPT-4 Turbo | 128K tokens |
| 2024 | Gemini 1.5 Pro | 1M tokens / 2M tokens |
| 2024 | Claude 3 | 200K tokens |

This progression has been enabled by three parallel advances: RoPE extension techniques, efficient attention implementations (FlashAttention), and architectural innovations.

### Efficient Attention Mechanisms

The quadratic bottleneck has spawned a rich ecosystem of efficient attention variants:

- **FlashAttention** (Dao et al., 2022): Does not reduce computational complexity but dramatically reduces memory usage by computing attention blockwise without materializing the full $n \times n$ attention matrix. This is an IO-aware optimization, not an approximation.
- **Multi-Query Attention (MQA)**: Shares keys and values across all attention heads, reducing KV cache memory by a factor equal to the number of heads.
- **Grouped-Query Attention (GQA)**: A compromise between full multi-head attention and MQA, grouping heads to share KV pairs. Used by LLaMA 2 70B, Mistral, and many modern models.
- **Ring Attention**: Distributes long sequences across multiple devices, with each device computing attention for its block while passing KV blocks in a ring topology. Enables near-unlimited context by adding devices.
- **Sparse Attention**: Allows each token to attend only to a subset of other tokens (e.g., local window plus global tokens), reducing complexity to $O(n\sqrt{n})$ or $O(n \log n)$.

### The "Lost in the Middle" Phenomenon

A critical finding from Liu et al. (2023) is that LLMs do not use their context window uniformly. When relevant information is placed at the beginning or end of a long context, models retrieve it well. When it's in the middle, performance degrades significantly.

This creates a U-shaped attention curve: strong attention at the start (primacy bias, possibly amplified by the BOS token's absolute position) and at the end (recency bias, natural for autoregressive models), with a valley in the middle.

This means the **effective context** -- the portion of the window the model reliably uses -- can be substantially smaller than the **nominal context length**.

### Context Window vs. Effective Context

These are different concepts:

- **Nominal context length**: The maximum number of tokens the model can accept as input (e.g., 128K for GPT-4 Turbo).
- **Effective context length**: The range within which the model reliably attends to and uses information. This is measured by "needle-in-a-haystack" tests, where a specific fact is hidden at various positions in a long document and the model is queried about it.

Models have improved on this metric over time. Claude 3 and Gemini 1.5 Pro showed strong needle-in-a-haystack performance across nearly their full context, but "nearly" is doing real work -- edge cases and multi-hop reasoning across long context remain challenging.

## Why It Matters

The context window is one of the most practically important properties of an LLM:

- **Document analysis**: A 128K context window can hold approximately a 300-page book. A 4K window holds roughly 6 pages. This determines whether you can ask questions about an entire codebase or just a single file.
- **Conversation memory**: In chatbot applications, the context window IS the model's memory. When the conversation exceeds the window, older messages must be dropped. This is why chatbots "forget" things from earlier in a conversation.
- **Agentic workflows**: LLM agents that iterate on tasks accumulate context rapidly. Tool calls, observations, and reasoning traces can consume the entire context window within a few dozen steps.
- **Cost**: Longer context means more tokens processed, and API costs scale with token count. Processing 128K tokens is ~32x more expensive than processing 4K tokens, all else being equal.
- **Latency**: Time-to-first-token increases with context length because the entire prompt must be processed before generation begins. The prefill time scales with context length (approximately linearly with FlashAttention, quadratically without).

## Key Technical Details

- The KV cache stores key and value tensors for all previous tokens across all layers during autoregressive generation. For a 70B parameter model with 128K context, the KV cache alone can exceed 40 GB of memory.
- Context window limits apply to **input + output** combined. If the model has a 128K context window and your input is 120K tokens, the model can only generate ~8K tokens of output.
- Sliding window attention (used by Mistral 7B) limits each token to attending to only the previous $W$ tokens (e.g., $W = 4096$). Information beyond the window can still propagate through multiple layers, but with degradation. This is a $O(n \cdot W)$ approach.
- Prefix caching allows reuse of computed KV caches across requests that share a common prefix (e.g., the same system prompt), dramatically reducing latency for applications with repeated long context.
- Context window expansion through RoPE modification (PI, NTK, YaRN) typically requires continued pretraining or fine-tuning on long sequences.

## Common Misconceptions

- **"A model with 128K context remembers everything in 128K tokens equally well."** The "lost in the middle" phenomenon shows this is false. Position of information matters.
- **"Longer context is always better."** Longer context adds latency, cost, and can introduce irrelevant information that distracts the model. For many tasks, well-curated short context outperforms dumping everything into a long context.
- **"Context window and training data length are the same."** Models can be trained on short sequences and extended at inference time through positional interpolation, though this may require fine-tuning.
- **"RAG is just a workaround for small context windows."** Even with million-token context, RAG remains valuable because retrieval can be more precise than relying on the model to find relevant passages in a sea of tokens, and it avoids the cost of processing irrelevant context.
- **"Running out of context makes the model stop."** When output hits the context limit, the model generates an EOS token or is truncated. It doesn't crash -- it just stops, potentially mid-sentence.

## Connections to Other Concepts

- **Positional Encoding / RoPE**: The positional encoding scheme determines the model's native context length and its ability to extend beyond training length.
- **Tokenization**: Tokenization efficiency directly converts between "words of text" and "tokens consumed," determining how much human-readable content fits in the window.
- **Token Embeddings**: Each position in the context window is occupied by an embedding vector; the full context is a matrix of these vectors.
- **Special Tokens**: The BOS token anchors position 0, and the EOS token signals the end of useful context.
- **Self-Attention**: The quadratic cost of attention with respect to sequence length is the fundamental computational constraint on context window size.

## Diagrams and Visualizations

![The "Lost in the Middle" U-shaped curve showing how LLM accuracy varies depending on where relevant information is placed within the context window](https://github.com/nelson-liu/lost-in-the-middle/raw/main/lost-in-the-middle.png)
*Source: [Nelson Liu – Lost in the Middle GitHub Repository](https://github.com/nelson-liu/lost-in-the-middle)*

![Illustration of the self-attention mechanism showing quadratic scaling of computation with sequence length, the fundamental constraint on context window size](https://jalammar.github.io/images/t/transformer_self-attention_visualization.png)
*Source: [Jay Alammar – The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)*

*See also the Needle-in-a-Haystack evaluation visualizations at: [Greg Kamradt's NIAH Testing](https://github.com/gkamradt/LLMTest_NeedleInAHaystack) -- heatmaps showing model retrieval accuracy across different context depths and positions.*

## Further Reading

- Liu, N.F., et al. (2023). "Lost in the Middle: How Language Models Use Long Contexts." *arXiv:2307.03172.* -- The landmark paper documenting the U-shaped attention phenomenon.
- Dao, T., et al. (2022). "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness." *NeurIPS 2022.* -- The breakthrough in efficient attention computation that enabled practical long-context models.
- Reid, M., et al. (2024). "Gemini 1.5: Unlocking multimodal understanding across millions of tokens of context." *arXiv:2403.05530.* -- Documents the push to million-token context with near-perfect needle-in-a-haystack retrieval.
