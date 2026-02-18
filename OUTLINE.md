# Complete Concept Index

All 153 concepts organized by category. Each entry links to its deep-dive file and includes a one-line summary.

---

## 01 - Foundational Architecture (20 concepts)

| Concept | Summary |
|---------|---------|
| [Activation Functions](01-foundational-architecture/activation-functions.md) | The evolution from ReLU to GELU to SwiGLU -- smoother, gated functions that improve LLM training dynamics. |
| [Attention Sinks](01-foundational-architecture/attention-sinks.md) | First tokens accumulate disproportionate attention scores regardless of content; exploited by StreamingLLM for infinite-length inference. |
| [Autoregressive Generation](01-foundational-architecture/autoregressive-generation.md) | LLMs produce text one token at a time, feeding each output back as input -- the source of both generative power and inference bottlenecks. |
| [Byte Latent Transformers](01-foundational-architecture/byte-latent-transformers.md) | Tokenizer-free architecture operating directly on raw bytes with dynamic patching, eliminating tokenization artifacts. |
| [Causal Attention](01-foundational-architecture/causal-attention.md) | Triangular mask restricting each token to attend only to preceding tokens, enforcing left-to-right autoregressive generation. |
| [Differential Transformer](01-foundational-architecture/differential-transformer.md) | Computes attention as the difference between two softmax maps, canceling noise like a differential amplifier. |
| [Encoder-Decoder Architecture](01-foundational-architecture/encoder-decoder-architecture.md) | The three Transformer paradigms (encoder-only, decoder-only, encoder-decoder) and why decoder-only dominates. |
| [Feed-Forward Networks](01-foundational-architecture/feed-forward-networks.md) | Two-layer MLP in each Transformer layer acting as the model's primary knowledge store (~2/3 of parameters). |
| [Grouped Query Attention](01-foundational-architecture/grouped-query-attention.md) | Shares KV heads across groups of query heads, achieving near-full-attention quality at a fraction of memory cost. |
| [Layer Normalization](01-foundational-architecture/layer-normalization.md) | Standardizes activations across the feature dimension, stabilizing deep Transformer training. |
| [Logits and Softmax](01-foundational-architecture/logits-and-softmax.md) | Raw output scores converted to a probability distribution for next-token selection. |
| [Mixture of Depths](01-foundational-architecture/mixture-of-depths.md) | Dynamically routes tokens through full computation or skip connections, reducing FLOPs by up to 50%. |
| [Mixture of Experts](01-foundational-architecture/mixture-of-experts.md) | Replaces dense FFN with multiple parallel experts and a router, scaling parameters while keeping per-token compute constant. |
| [Multi-Head Attention](01-foundational-architecture/multi-head-attention.md) | Parallel attention operations each capturing different relationship types (syntactic, semantic, positional). |
| [Next-Token Prediction](01-foundational-architecture/next-token-prediction.md) | The deceptively simple training objective that gives rise to emergent capabilities at scale. |
| [Residual Connections](01-foundational-architecture/residual-connections.md) | Skip connections creating a "residual stream" that enables training of networks with hundreds of layers. |
| [Self-Attention](01-foundational-architecture/self-attention.md) | Every token dynamically computes a weighted combination of all other tokens' representations. |
| [Sliding Window Attention](01-foundational-architecture/sliding-window-attention.md) | Local attention window of W tokens reduces quadratic cost to linear; layer stacking extends effective receptive field. |
| [Sparse Attention](01-foundational-architecture/sparse-attention.md) | Restricts attention to token subsets, reducing O(n^2) to O(n log n) or O(n) for long sequences. |
| [Transformer Architecture](01-foundational-architecture/transformer-architecture.md) | The attention-based neural network architecture that is the universal foundation of modern LLMs. |

---

## 02 - Input Representation (9 concepts)

| Concept | Summary |
|---------|---------|
| [ALiBi](02-input-representation/alibi.md) | Linear biases on attention scores replacing learned positional embeddings, enabling length extrapolation with zero parameters. |
| [Byte-Pair Encoding](02-input-representation/byte-pair-encoding.md) | Compression algorithm repurposed for tokenization that iteratively merges frequent adjacent symbol pairs. |
| [Context Window](02-input-representation/context-window.md) | The fixed-length token span a transformer can attend to -- the model's "working memory." |
| [Positional Encoding](02-input-representation/positional-encoding.md) | Injecting token order information into an architecture that would otherwise treat input as unordered. |
| [Rotary Position Embedding](02-input-representation/rotary-position-embedding.md) | Encodes positions by rotating Q/K vectors so dot products naturally depend on relative distance. |
| [Special Tokens](02-input-representation/special-tokens.md) | Reserved vocabulary entries carrying control signals (sequence boundaries, turn-taking) rather than content. |
| [Token Embeddings](02-input-representation/token-embeddings.md) | Converting discrete token IDs into dense vectors where geometry encodes semantics. |
| [Tokenization](02-input-representation/tokenization.md) | Breaking raw text into discrete units for numerical processing -- choices here ripple through everything. |
| [Vocabulary Design](02-input-representation/vocabulary-design.md) | Choosing vocabulary size and composition, balancing compression, embedding size, and multilingual fairness. |

---

## 03 - Training Fundamentals (16 concepts)

| Concept | Summary |
|---------|---------|
| [Adam/AdamW Optimizer](03-training-fundamentals/adam-optimizer.md) | The near-universal LLM optimizer combining adaptive learning rates, momentum, and decoupled weight decay. |
| [Backpropagation](03-training-fundamentals/backpropagation.md) | Computing per-parameter error contributions to enable gradient descent across billions of parameters. |
| [Catastrophic Forgetting](03-training-fundamentals/catastrophic-forgetting.md) | Abrupt loss of old knowledge when training on new tasks -- gradient updates overwrite critical parameters. |
| [Cross-Entropy Loss](03-training-fundamentals/cross-entropy-loss.md) | The objective function measuring how "surprised" the model is by the actual next token. |
| [Curriculum Learning](03-training-fundamentals/curriculum-learning.md) | Training in meaningful order (easy to hard) for better convergence and final performance at same compute. |
| [Emergent Abilities](03-training-fundamentals/emergent-abilities.md) | Capabilities that appear suddenly at certain scale thresholds -- exciting but hard to forecast. |
| [Gradient Checkpointing](03-training-fundamentals/gradient-checkpointing.md) | Trading recomputation for memory savings by selectively storing activations during training. |
| [Gradient Clipping & Accumulation](03-training-fundamentals/gradient-clipping.md) | Three stability techniques: clipping prevents explosions, accumulation simulates larger batches, checkpointing saves memory. |
| [Grokking](03-training-fundamentals/grokking.md) | Sudden generalization long after memorization -- challenging assumptions about when models truly learn. |
| [Learning Rate Scheduling](03-training-fundamentals/learning-rate-scheduling.md) | Warmup then decay to prevent instability and ensure convergence to good minima. |
| [Mixed Precision Training](03-training-fundamentals/mixed-precision-training.md) | FP16/BF16 computation with FP32 master weights -- halves memory and doubles throughput via tensor cores. |
| [Model Collapse](03-training-fundamentals/model-collapse.md) | Progressive quality degradation from recursive training on AI-generated data. |
| [Pre-Training](03-training-fundamentals/pre-training.md) | The foundational phase where models learn language, facts, and reasoning from trillions of tokens. |
| [Scaling Laws](03-training-fundamentals/scaling-laws.md) | Power-law relationships enabling prediction of model performance before spending hundreds of millions. |
| [Self-Play & Self-Improvement](03-training-fundamentals/self-play-and-self-improvement.md) | Bootstrapping stronger capabilities from own outputs -- STaR, SPIN, ReST approaches. |
| [Training Data Curation](03-training-fundamentals/training-data-curation.md) | Collecting, filtering, deduplicating, and mixing datasets -- quality consistently beats quantity. |

---

## 04 - Distributed Training (7 concepts)

| Concept | Summary |
|---------|---------|
| [3D Parallelism](04-distributed-training/3d-parallelism.md) | Combining data, tensor, and pipeline parallelism mapped to hardware topology for training the largest models. |
| [Data Parallelism](04-distributed-training/data-parallelism.md) | Replicating the model on every GPU, splitting data, and synchronizing gradients. |
| [Expert Parallelism](04-distributed-training/expert-parallelism.md) | Distributing MoE experts across GPUs with all-to-all communication for trillion-parameter models. |
| [Pipeline Parallelism](04-distributed-training/pipeline-parallelism.md) | Distributing consecutive layers across GPUs with micro-batching to minimize pipeline bubbles. |
| [Ring Attention](04-distributed-training/ring-attention.md) | Distributing long sequences across GPUs in a ring topology with overlapped communication and computation. |
| [Tensor Parallelism](04-distributed-training/tensor-parallelism.md) | Splitting individual layers across GPUs so each computes a slice of every layer's output. |
| [ZeRO & FSDP](04-distributed-training/zero-and-fsdp.md) | Eliminating data-parallel memory redundancy by sharding optimizer states, gradients, and parameters. |

---

## 05 - Alignment & Post-Training (13 concepts)

| Concept | Summary |
|---------|---------|
| [Chain-of-Thought Training](05-alignment-and-post-training/chain-of-thought-training.md) | From prompting trick to training paradigm -- models explicitly trained to reason before answering. |
| [Constitutional AI](05-alignment-and-post-training/constitutional-ai.md) | Replacing human labels with AI feedback guided by explicit principles for scalable alignment. |
| [DPO](05-alignment-and-post-training/dpo.md) | Collapsing RLHF into a single supervised step by deriving optimal policy directly from preferences. |
| [GRPO](05-alignment-and-post-training/grpo.md) | DeepSeek's critic-free RL algorithm using group-based relative scoring of sampled outputs. |
| [Preference Learning Variants](05-alignment-and-post-training/preference-learning-variants.md) | DPO alternatives (IPO, KTO, ORPO) with different data requirements and robustness trade-offs. |
| [Process Reward Models](05-alignment-and-post-training/process-reward-models.md) | Evaluating each reasoning step vs. just the final answer -- from "right answer?" to "right reasoning?" |
| [Rejection Sampling](05-alignment-and-post-training/rejection-sampling.md) | Best-of-N sampling scored by a reward model -- captured most alignment gains in Llama 2. |
| [Reward Modeling](05-alignment-and-post-training/reward-modeling.md) | Training a neural network to predict human preferences -- the single biggest alignment bottleneck. |
| [RLAIF](05-alignment-and-post-training/rlaif.md) | AI-generated preference labels matching human quality at ~$0.001 per comparison vs. $1-10 for humans. |
| [RLHF](05-alignment-and-post-training/rlhf.md) | The foundational alignment technique: reward model from human comparisons + RL optimization + KL penalty. |
| [RLVR](05-alignment-and-post-training/rlvr.md) | RL with objectively verifiable rewards (math correctness, code tests) avoiding Goodhart's Law. |
| [Supervised Fine-Tuning](05-alignment-and-post-training/supervised-fine-tuning.md) | Transforming a next-token predictor into an instruction-following assistant via (instruction, response) pairs. |
| [Synthetic Data](05-alignment-and-post-training/synthetic-data.md) | Using LLMs to generate training data -- scalable but risks model collapse and inherited biases. |

---

## 06 - Parameter-Efficient Fine-Tuning (5 concepts)

| Concept | Summary |
|---------|---------|
| [Adapters & Prompt Tuning](06-parameter-efficient-fine-tuning/adapters-and-prompt-tuning.md) | Bottleneck adapters, prefix tuning, prompt tuning, (IA)^3, DoRA -- each with distinct trade-offs. |
| [Full vs. PEFT Fine-Tuning](06-parameter-efficient-fine-tuning/full-vs-peft-fine-tuning.md) | Full fine-tuning vs. parameter-efficient methods -- the gap vanishes at sufficient scale. |
| [LoRA](06-parameter-efficient-fine-tuning/lora.md) | Frozen weights + small trainable low-rank matrices achieving fine-tuning quality at a fraction of parameters. |
| [Multi-LoRA Serving](06-parameter-efficient-fine-tuning/multi-lora-serving.md) | Serving thousands of LoRA adapters simultaneously from a single shared base model. |
| [QLoRA](06-parameter-efficient-fine-tuning/qlora.md) | 4-bit quantized base + LoRA adapters enabling 65B+ fine-tuning on a single 48GB GPU. |

---

## 07 - Inference & Deployment (18 concepts)

| Concept | Summary |
|---------|---------|
| [Constrained Decoding](07-inference-and-deployment/constrained-decoding.md) | Forcing output to conform to formal grammars (JSON, regex) by masking invalid tokens -- 100% structural guarantee. |
| [Continuous Batching](07-inference-and-deployment/continuous-batching.md) | Inserting/retiring sequences every decoding step for 10-23x throughput over static batching. |
| [Distillation for Reasoning](07-inference-and-deployment/distillation-for-reasoning.md) | Transferring chain-of-thought from large teachers to small students -- distillation outperforms direct RL at small scale. |
| [Flash Attention](07-inference-and-deployment/flash-attention.md) | IO-aware attention keeping data in fast SRAM, reducing memory to O(N) with 2-4x speedups -- exact, not approximate. |
| [Knowledge Distillation](07-inference-and-deployment/knowledge-distillation.md) | Student model learning from teacher's soft probability distributions, transferring rich inter-class knowledge. |
| [KV Cache](07-inference-and-deployment/kv-cache.md) | Storing computed key-value tensors to avoid recomputation, turning O(n^2) generation into O(n). |
| [KV Cache Compression](07-inference-and-deployment/kv-cache-compression.md) | Quantization, eviction, and merging reducing KV memory 2-8x for practical 128K+ context deployment. |
| [Medusa / Parallel Decoding](07-inference-and-deployment/medusa-parallel-decoding.md) | Lightweight prediction heads enabling parallel generation with 2-3x speedups, no draft model needed. |
| [Model Routing](07-inference-and-deployment/model-routing.md) | Dynamically selecting which LLM per query for 40-60% cost reduction while maintaining quality. |
| [Model Serving Frameworks](07-inference-and-deployment/model-serving.md) | Orchestrating weights, memory, batching, and delivery -- framework choice means 10-23x throughput difference. |
| [PagedAttention](07-inference-and-deployment/paged-attention.md) | OS-style virtual memory paging for KV cache, eliminating 60-80% memory waste for 2-4x throughput. |
| [Prefill-Decode Disaggregation](07-inference-and-deployment/prefill-decode-disaggregation.md) | Separating compute-bound prefill and memory-bound decode onto different hardware for 1.5-2x efficiency. |
| [Prefix Caching](07-inference-and-deployment/prefix-caching.md) | Storing KV states for shared prompt prefixes -- up to 90% cost savings and 85% TTFT reduction. |
| [Prompt Compression](07-inference-and-deployment/prompt-compression.md) | Perplexity-based token pruning cutting costs by 75% and accelerating prefill 2-4x. |
| [Quantization](07-inference-and-deployment/quantization.md) | Reducing precision to 8-bit or 4-bit, shrinking memory 2-4x with surprisingly small quality loss. |
| [Sampling Strategies](07-inference-and-deployment/sampling-strategies.md) | Temperature, Top-K, Top-P controlling the coherence-creativity trade-off in token selection. |
| [Speculative Decoding](07-inference-and-deployment/speculative-decoding.md) | Draft model guesses multiple tokens, target verifies all at once -- 2-3x speedup, mathematically identical output. |
| [Throughput vs. Latency](07-inference-and-deployment/throughput-vs-latency.md) | The fundamental competing objectives in LLM serving and the architecture decisions they drive. |

---

## 08 - Practical Applications (11 concepts)

| Concept | Summary |
|---------|---------|
| [AI Agents](08-practical-applications/ai-agents.md) | LLMs operating in autonomous loops: reasoning, acting through tools, observing, and iterating. |
| [Chunking Strategies](08-practical-applications/chunking-strategies.md) | How you split documents for RAG determines whether retrieval returns useful context or useless fragments. |
| [Embedding Models & Vector DBs](08-practical-applications/embedding-models-and-vector-databases.md) | Text-to-vector transformation and similarity search at scale -- the retrieval backbone of LLM apps. |
| [Function Calling & Tool Use](08-practical-applications/function-calling-and-tool-use.md) | Structured JSON requests turning LLMs from text generators into reasoning engines that take real actions. |
| [Memory Systems](08-practical-applications/memory-systems.md) | Extending agents beyond context windows with structured storage, retrieval, and management across sessions. |
| [Multi-Agent Systems](08-practical-applications/multi-agent-systems.md) | Multiple LLM agents collaborating through defined roles, tools, and communication protocols. |
| [Prompt Engineering](08-practical-applications/prompt-engineering.md) | Crafting inputs that reliably elicit desired outputs -- bridging model capability and practical utility. |
| [RAG](08-practical-applications/rag.md) | Grounding responses in retrieved external knowledge to reduce hallucination and answer beyond training data. |
| [ReAct Pattern](08-practical-applications/react-pattern.md) | Thought-Action-Observation loops interleaving reasoning with tool use for grounded agent behavior. |
| [Self-Reflection](08-practical-applications/self-reflection.md) | Agents evaluating and iteratively improving their own outputs using natural language memory. |
| [Structured Output](08-practical-applications/structured-output.md) | Constraining generation to parseable formats (JSON, XML) for reliable software integration. |

---

## 09 - Safety & Alignment (21 concepts)

| Concept | Summary |
|---------|---------|
| [Adversarial Robustness](09-safety-and-alignment/adversarial-robustness.md) | Attacks (GCG, AutoDAN, PAIR) exploiting model vulnerabilities and defenses against them -- attackers hold structural advantage. |
| [AI Sandbagging](09-safety-and-alignment/ai-sandbagging.md) | Models strategically underperforming on evaluations to avoid triggering safety restrictions. |
| [Alignment Problem](09-safety-and-alignment/alignment-problem.md) | Ensuring AI pursues intended goals rather than optimizing proxy objectives that diverge from human values. |
| [Bias & Fairness](09-safety-and-alignment/bias-and-fairness.md) | LLMs absorb and amplify training data biases -- full elimination may be fundamentally impossible. |
| [Circuit Breakers](09-safety-and-alignment/circuit-breakers.md) | Representation-level safety redirecting harmful activations, far more robust than RLHF-based refusal. |
| [Goodhart's Law](09-safety-and-alignment/goodharts-law.md) | When a measure becomes a target, it ceases to be good -- the root cause of reward hacking. |
| [Guardrails](09-safety-and-alignment/guardrails.md) | Multi-layered defense systems preventing harmful, off-topic, or policy-violating outputs. |
| [Hallucination](09-safety-and-alignment/hallucination.md) | Confident but wrong outputs because models optimize for plausibility, not truth. |
| [Instruction Hierarchy](09-safety-and-alignment/instruction-hierarchy.md) | Strict priority levels (system > developer > user) defending against prompt injection. |
| [Jailbreaking](09-safety-and-alignment/jailbreaking.md) | Adversarial techniques circumventing safety guardrails -- exposing capability-safety tensions. |
| [Machine Unlearning](09-safety-and-alignment/machine-unlearning.md) | Selectively removing training data influence without retraining -- for GDPR, copyright, and safety. |
| [Prompt Injection](09-safety-and-alignment/prompt-injection.md) | Attacker inputs overriding system behavior -- may be fundamentally unsolvable due to shared instruction/data channel. |
| [Red Teaming](09-safety-and-alignment/red-teaming.md) | Adversarial testing to discover failures before users encounter them in production. |
| [Reward Hacking](09-safety-and-alignment/reward-hacking.md) | Exploiting unintended reward function shortcuts to maximize score without achieving intended objectives. |
| [Scalable Oversight](09-safety-and-alignment/scalable-oversight.md) | Maintaining human control as AI exceeds supervisor capability -- debate, amplification, process supervision. |
| [Sleeper Agents](09-safety-and-alignment/sleeper-agents.md) | Hidden conditional behaviors surviving safety training -- empirical proof that RLHF cannot remove backdoors. |
| [Specification Gaming](09-safety-and-alignment/specification-gaming.md) | Satisfying the letter of the objective while violating the spirit -- the central alignment challenge. |
| [Sycophancy](09-safety-and-alignment/sycophancy.md) | RLHF-trained models agreeing with wrong users -- optimizing for approval over truthfulness. |
| [Toxicity Detection](09-safety-and-alignment/toxicity-detection.md) | Identifying harmful content while navigating the boundary between sensitive topics and genuine harm. |
| [Watermarking](09-safety-and-alignment/watermarking-llm-text.md) | Embedding detectable but imperceptible signals in generated text for AI content identification. |
| [Weak-to-Strong Generalization](09-safety-and-alignment/weak-to-strong-generalization.md) | Whether weaker systems can effectively supervise stronger ones -- the core superalignment question. |

---

## 10 - Evaluation (7 concepts)

| Concept | Summary |
|---------|---------|
| [Benchmark Contamination](10-evaluation/benchmark-contamination-detection.md) | Detecting whether models trained on test data -- n-gram overlap, Min-K% Prob, canary strings, perplexity tests. |
| [Benchmarks](10-evaluation/benchmarks.md) | Standardized test suites for measuring LLM capabilities -- the primary (if imperfect) comparison basis. |
| [Chatbot Arena](10-evaluation/chatbot-arena.md) | LMSYS crowdsourced head-to-head evaluation with 2M+ votes -- the most trusted LLM ranking. |
| [Evaluation Metrics](10-evaluation/evaluation-metrics.md) | BLEU, ROUGE, BERTScore -- automated metrics with known strengths and limitations. |
| [Human Evaluation](10-evaluation/human-evaluation.md) | The gold standard via pairwise preference and ELO ranking, threatened by benchmark contamination. |
| [LLM-as-Judge](10-evaluation/llm-as-judge.md) | Using strong LLMs to evaluate other LLMs -- scalable but introduces systematic biases. |
| [Perplexity](10-evaluation/perplexity.md) | How "surprised" a model is by new text -- the most fundamental intrinsic evaluation metric. |

---

## 11 - Advanced & Emerging (26 concepts)

| Concept | Summary |
|---------|---------|
| [Agentic RAG](11-advanced-and-emerging/agentic-rag.md) | Replacing rigid retrieve-then-generate with an agent that dynamically reasons about retrieval. |
| [ColBERT](11-advanced-and-emerging/colbert-late-interaction.md) | Multi-vector late interaction achieving cross-encoder accuracy at bi-encoder speed via MaxSim. |
| [Compound AI Systems](11-advanced-and-emerging/compound-ai-systems.md) | Combining LLMs with retrievers, tools, and verifiers -- "better systems" over "better models." |
| [Context Window Extension](11-advanced-and-emerging/context-window-extension.md) | Techniques stretching context from 512 to 1M+ tokens via positional encoding and architecture tricks. |
| [Corrective RAG](11-advanced-and-emerging/corrective-rag.md) | Evaluating retrieval quality and taking corrective action (rewriting, web fallback) when it fails. |
| [GraphRAG](11-advanced-and-emerging/graphrag.md) | Knowledge graph construction with hierarchical community summaries for global sensemaking queries. |
| [HyDE](11-advanced-and-emerging/hyde-hypothetical-document-embeddings.md) | Generating hypothetical answer documents as retrieval vectors -- closer to real answers than the question. |
| [In-Context Learning](11-advanced-and-emerging/in-context-learning.md) | Learning new tasks from prompt examples at inference time, without gradient updates. |
| [Inference-Time Scaling Laws](11-advanced-and-emerging/inference-time-scaling-laws.md) | More inference compute (sampling, search, verification) predictably improves reasoning performance. |
| [Late Chunking](11-advanced-and-emerging/late-chunking.md) | Embed full document first, then chunk -- preserving cross-chunk context that traditional chunking destroys. |
| [Matryoshka Embeddings](11-advanced-and-emerging/matryoshka-representation-learning.md) | Any prefix of an embedding is itself valid -- one model, multiple dimensionalities, graceful degradation. |
| [Mechanistic Interpretability](11-advanced-and-emerging/mechanistic-interpretability.md) | Reverse-engineering neural networks at the circuit level to understand features, computations, and behaviors. |
| [Model Merging](11-advanced-and-emerging/model-merging.md) | Combining weights from separately trained models without additional training -- exploiting loss landscape geometry. |
| [Multi-Token Prediction](11-advanced-and-emerging/multi-token-prediction.md) | Predicting several future tokens simultaneously for richer representations and faster inference. |
| [Multimodal Models](11-advanced-and-emerging/multimodal-models.md) | Extending LLMs with vision, audio, and other modality encoders for cross-modal reasoning. |
| [Neurosymbolic AI](11-advanced-and-emerging/neurosymbolic-ai.md) | Combining neural pattern recognition with symbolic precision for formally verifiable reasoning. |
| [Query Decomposition](11-advanced-and-emerging/query-decomposition-and-multi-step-retrieval.md) | Breaking complex queries into sub-queries for multi-hop retrieval that single-shot cannot handle. |
| [RAPTOR](11-advanced-and-emerging/raptor.md) | Recursive tree index with clustered summaries enabling retrieval at any level of abstraction. |
| [Reasoning Models](11-advanced-and-emerging/reasoning-models.md) | Extended internal deliberation (o1/R1) trading inference compute for dramatically improved accuracy. |
| [Representation Engineering](11-advanced-and-emerging/representation-engineering.md) | Controlling behavior by adding/subtracting steering vectors in activation space -- no weight updates needed. |
| [Reranking & Cross-Encoders](11-advanced-and-emerging/reranking-and-cross-encoders.md) | Two-stage retrieve-then-rerank architecture for production-grade retrieval precision. |
| [Self-RAG](11-advanced-and-emerging/self-rag.md) | Model decides when to retrieve, evaluates relevance, and judges its own generation via reflection tokens. |
| [State Space Models](11-advanced-and-emerging/state-space-models.md) | Linear-time sequence modeling via learned recurrent state updates -- Mamba as the credible Transformer alternative. |
| [Test-Time Compute](11-advanced-and-emerging/test-time-compute.md) | The paradigm shift from bigger models to harder thinking -- allocating compute at inference. |
| [Tree-of-Thought](11-advanced-and-emerging/tree-of-thought.md) | Exploring multiple reasoning paths with backtracking -- treating reasoning as search, not linear narrative. |
| [Vision-Language Models](11-advanced-and-emerging/vision-language-models.md) | Integrating visual perception with language understanding for AI that can see and reason. |
