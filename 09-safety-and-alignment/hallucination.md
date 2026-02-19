# Hallucination & Grounding

**One-Line Summary**: LLMs generate text that sounds confident and fluent but is sometimes factually wrong, because they were trained to produce *plausible* continuations, not *true* statements.

**Prerequisites**: Understanding of how language models are trained (next-token prediction), transformer architecture basics, and a general sense of how retrieval-augmented generation (RAG) works.

## What Is Hallucination?

Imagine a student who has read thousands of textbooks but never checked a single fact against the real world. When asked a question, they don't retrieve an answer from a database of verified truths -- they construct a response that *sounds like* what the right answer would sound like, based on the patterns they absorbed during study. Most of the time this works remarkably well. But sometimes the student confidently fabricates a citation that doesn't exist, invents a historical event that never happened, or subtly distorts a real fact into something slightly wrong.

*Recommended visual: Types of hallucination: intrinsic (contradicts source) vs extrinsic (unverifiable) — see [Survey of Hallucination in NLG (arXiv:2202.03629)](https://arxiv.org/abs/2202.03629)*


This is hallucination in large language models. The term refers to any generated output that is not grounded in the model's training data or provided context -- content that is fluent, confident, and entirely fabricated. It is not a bug in the traditional software sense; it is an emergent consequence of how these models fundamentally work.

The complementary concept is **grounding**: techniques that anchor a model's outputs to verifiable sources of truth, constraining its natural tendency to confabulate.

## How It Works


*Recommended visual: Hallucination mitigation strategies: retrieval augmentation, self-consistency, citation generation — see [Hugging Face Blog](https://huggingface.co/blog)*

### Why Hallucination Happens

LLMs are trained via next-token prediction: given a sequence of tokens, predict the most probable continuation. The training objective optimizes for **statistical plausibility**, not factual accuracy. This creates a fundamental gap:

1. **No internal knowledge base.** The model does not store facts in a structured, retrievable way. Knowledge is distributed across billions of parameters as statistical associations. When those associations are weak or conflicting, the model interpolates -- and sometimes interpolates incorrectly.

2. **Confidence without calibration.** The model assigns probabilities to tokens, but these probabilities reflect linguistic likelihood, not epistemic certainty. A hallucinated fact can receive higher probability than a true but unusual one simply because it fits the linguistic pattern better.

3. **Training data gaps and conflicts.** If a topic is underrepresented in training data, or if the training data contains contradictory information, the model may "fill in the gaps" by generating plausible-sounding content that has no factual basis.

4. **Decoding strategy effects.** Higher-temperature sampling and top-p sampling increase diversity but also increase hallucination risk. Greedy decoding reduces hallucination but produces repetitive, less useful text.

### Types of Hallucination

- **Intrinsic hallucination**: The output contradicts the provided source material. For example, a summarization model says a paper found X when the paper actually found Y. This is directly measurable against the source.

- **Extrinsic hallucination**: The output introduces claims that cannot be verified or refuted from the source material. The model adds information that may or may not be true but was never in the input. This is harder to detect because it requires external knowledge to evaluate.

- **Closed-domain hallucination**: Occurs when the model has a specific context (a document, a database result) and generates content unfaithful to that context. This is the primary concern in enterprise applications like document Q&A.

- **Open-domain hallucination**: Occurs in free-form generation where there is no specific grounding document. The model draws on parametric knowledge and may fabricate facts, citations, statistics, or events.

### Grounding Techniques

**Retrieval-Augmented Generation (RAG)** is the most widely adopted mitigation. The model is given retrieved documents as context and instructed to answer based only on that context. This converts open-domain hallucination into a closed-domain problem, which is more manageable. However, the model can still hallucinate relative to the retrieved context, and retrieval itself may return irrelevant or incorrect documents.

**Citation generation** requires the model to provide specific references for its claims. Systems like Bing Chat and Perplexity implement this by having the model output inline citations that map to retrieved sources. This improves verifiability but does not eliminate hallucination -- models sometimes generate citations that don't support the claim, or cite real sources inaccurately.

**Tool use and code execution** ground the model by offloading factual tasks to deterministic systems. Instead of computing "what is 7,392 times 4,518?" from parametric knowledge, the model calls a calculator. Instead of guessing the current weather, it calls a weather API. This eliminates hallucination for the tasks handled by tools.

**Chain-of-thought and self-consistency** methods ask the model to reason step-by-step, then sample multiple reasoning paths and check for consistency. If the model arrives at different answers via different reasoning chains, confidence is reduced. This helps but adds latency and cost.

## Why It Matters

Hallucination is widely regarded as the **number one barrier to enterprise adoption** of LLMs. In domains like healthcare, legal analysis, financial reporting, and education, a confident but wrong answer can have serious real-world consequences -- from misdiagnosis to legal liability.

The challenge is insidious because hallucinated text is, by definition, fluent and convincing. Unlike a traditional software error that crashes or returns an error code, a hallucination looks exactly like a correct answer. This means users must maintain constant vigilance, which undermines the productivity gains that LLMs promise.

For content creation, hallucination erodes trust. A single fabricated citation in an otherwise excellent article can destroy credibility. For code generation, hallucinated API calls or non-existent library functions cause bugs that are hard to diagnose because the code *looks* correct.

## Key Technical Details

- Hallucination rates vary dramatically by domain: models hallucinate less on well-represented topics (popular programming languages, common historical events) and more on niche or recent topics.
- Larger models generally hallucinate less than smaller ones, but no model achieves zero hallucination. Scaling alone does not solve the problem.
- RAG reduces but does not eliminate hallucination. Studies show RAG-augmented systems still hallucinate 3-15% of the time depending on the domain and evaluation criteria.
- Faithfulness evaluation metrics include **ROUGE-L** (surface-level overlap), **BERTScore** (semantic similarity), **QuestEval** (question-based verification), and human-annotated **attribution scores** that check whether each claim is supported by the cited source.
- Fine-tuning on high-quality, factual data and reinforcement learning from human feedback (RLHF) both reduce hallucination rates but introduce their own trade-offs (reduced creativity, increased refusals).
- Temperature and sampling parameters have a direct, measurable impact on hallucination frequency. Production systems typically use lower temperatures (0.0-0.3) for factual tasks.

## Common Misconceptions

- **"Hallucination means the model is lying."** The model has no concept of truth or deception. It generates probable token sequences. Hallucination is a statistical artifact, not intentional fabrication.
- **"RAG solves hallucination."** RAG significantly reduces it by providing grounding context, but models can still ignore, misinterpret, or go beyond the retrieved context. RAG changes the problem; it does not eliminate it.
- **"Bigger models don't hallucinate."** They hallucinate less frequently, but they can also hallucinate more convincingly, making the errors harder to catch. Size helps but is not a complete solution.
- **"You can detect hallucination by checking the model's confidence."** Token-level probabilities are not well-calibrated indicators of factual accuracy. A model can assign high probability to a hallucinated claim.
- **"Hallucination is always bad."** In creative writing, brainstorming, and ideation, the model's ability to generate novel combinations -- which is mechanistically the same as hallucination -- is a feature, not a bug. The problem is context-dependent.

## Connections to Other Concepts

- **Retrieval-Augmented Generation (RAG)**: The primary architectural mitigation for hallucination. RAG's effectiveness is measured largely by how much it reduces hallucination.
- **RLHF and Alignment**: RLHF training can teach models to say "I don't know" rather than hallucinate, trading coverage for accuracy.
- **Prompt Engineering**: Careful prompt design (e.g., "Answer only based on the provided context. If you cannot answer, say so.") reduces hallucination in practice.
- **Evaluation and Benchmarks**: Hallucination measurement is a core component of LLM evaluation, with benchmarks like TruthfulQA specifically targeting it.
- **Guardrails**: Post-generation fact-checking and citation-verification systems serve as a safety net against hallucinated outputs reaching end users.

## Further Reading

- Huang et al., "A Survey on Hallucination in Large Language Models" (2023) -- Comprehensive taxonomy and survey of hallucination types, causes, and mitigations.
- Min et al., "FActScore: Fine-grained Atomic Evaluation of Factual Precision in Long Form Text Generation" (2023) -- Introduces a method for decomposing generated text into atomic facts and verifying each against a knowledge source.
- Shuster et al., "Retrieval Augmentation Reduces Hallucination in Conversation" (2021) -- Foundational work demonstrating that retrieval-augmented models produce significantly fewer hallucinations than purely parametric models.
