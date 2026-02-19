# Multimodal Models

**One-Line Summary**: Multimodal models extend LLMs beyond text by connecting vision encoders, audio processors, and other modality-specific modules to a language model backbone, enabling AI systems that can see, hear, and reason across different types of input simultaneously.

**Prerequisites**: Understanding of Transformer architecture, self-attention and cross-attention mechanisms, pre-training and fine-tuning paradigms, embedding spaces, and contrastive learning basics.

## What Are Multimodal Models?

Imagine a person who can only read -- they are incredibly literate but blind and deaf. Now imagine giving them eyes and ears, but with a twist: everything they see and hear must first be translated into written descriptions before they can think about it. That is essentially how most multimodal LLMs work. The language model is the "brain" that reasons in text, and specialized encoders act as sensory organs that convert images, audio, and video into token-like representations the language model can process.

Multimodal models are AI systems that can process and reason about multiple types of input -- text, images, audio, video, and potentially more. The central challenge is **alignment**: ensuring that visual features, audio features, and text features all live in a shared representational space where the language model can reason about them coherently.

## How It Works

### Vision Encoders: Teaching Models to See

The most common approach to giving LLMs vision uses a pre-trained vision encoder to convert images into a sequence of visual tokens.

**Vision Transformer (ViT)**: Splits an image into fixed-size patches (typically 14x14 or 16x16 pixels), linearly projects each patch into an embedding, adds positional encodings, and processes the sequence through Transformer layers. The output is a grid of visual feature vectors.

**CLIP (Contrastive Language-Image Pre-training)**: OpenAI trained a ViT and a text encoder jointly on 400M image-text pairs using contrastive learning:

```
loss = -log(exp(sim(image_i, text_i) / tau) / sum_j(exp(sim(image_i, text_j) / tau)))
```

Where sim is cosine similarity and tau is a learned temperature. This produces visual features that are already partially aligned with language -- a crucial property for multimodal models.

**SigLIP (Sigmoid Loss for Language-Image Pre-training)**: Google's refinement replaces the softmax-based contrastive loss with a simpler sigmoid loss applied to each pair independently:

```
loss = -log(sigmoid(y * (sim(image, text) / tau - bias)))
```

Where y = +1 for matching pairs and y = -1 for non-matching pairs. This eliminates the need for large batch sizes (which CLIP requires for good negatives) and scales more efficiently.

### The Alignment Module: Bridging Vision and Language

Raw vision encoder outputs are not directly compatible with the language model's token embedding space. An alignment module bridges this gap:

- **Linear Projection**: The simplest approach -- a single linear layer maps visual features to the language model's embedding dimension. LLaVA-1.0 used this, demonstrating that a simple projection can be surprisingly effective.
- **MLP Projection**: LLaVA-1.5 upgraded to a two-layer MLP with GELU activation, improving feature alignment.
- **Q-Former (Querying Transformer)**: BLIP-2 uses a small Transformer with learned query tokens that cross-attend to visual features, compressing the visual representation into a fixed number of tokens (typically 32-64) regardless of image resolution.
- **Perceiver Resampler**: Flamingo uses a similar approach with learned latent queries that attend to visual features, producing a fixed-length visual prefix.

### Architecture Patterns

**Encoder-Projector-Decoder (Most Common)**:
```
Image -> Vision Encoder -> Projection -> [Visual Tokens + Text Tokens] -> LLM -> Output
```

The visual tokens are concatenated with or interleaved among text tokens, and the LLM processes them together. GPT-4V, Claude's vision, and LLaVA all follow this pattern.

**Cross-Attention Injection**: Instead of concatenating visual tokens into the input sequence, visual features are injected via cross-attention layers inserted into the LLM. Flamingo pioneered this: frozen gated cross-attention layers attend to visual features at regular intervals in the language model.

**Native Multimodal Training**: Rather than bolting a vision encoder onto a pre-trained LLM, some models (Gemini, Fuyu) are trained from scratch on interleaved multimodal data. The tokenizer itself handles images (e.g., Fuyu patches images directly into the Transformer without a separate encoder). This approach avoids the information bottleneck of a separate encoder but requires vastly more training data and compute.

### Audio and Video

**Audio Models**: Follow a similar pattern. Whisper-style audio encoders convert spectrograms into feature sequences. These are projected into the LLM's space. Models like GPT-4o process audio natively, enabling real-time voice conversation without a speech-to-text intermediate step.

**Video Understanding**: Typically samples frames from video, encodes each frame with the vision encoder, and adds temporal position encodings. The challenge is efficiency -- even a short video produces thousands of visual tokens. Approaches include temporal pooling, selecting keyframes, and using 3D convolutional encoders that capture motion.

## Why It Matters

Multimodal understanding is essential for AI systems that interact with the real world. Text alone cannot describe a medical scan, interpret a chart, debug a UI screenshot, or understand a gesture. Multimodal models enable:

- **Document understanding**: Processing PDFs, slides, and handwritten notes with layout awareness
- **Visual reasoning**: Answering questions that require understanding spatial relationships, counting, or reading text in images
- **Accessibility**: Describing images for visually impaired users, captioning video
- **Robotics and embodied AI**: Grounding language instructions in visual perception
- **Scientific analysis**: Interpreting microscopy images, satellite data, molecular structures

The trajectory is clear: future AI systems will be natively multimodal, processing all modalities as naturally as current LLMs process text.

## Key Technical Details

- **Resolution matters**: Higher-resolution images contain more detail but produce more visual tokens. LLaVA-NeXT and other models use dynamic resolution strategies, splitting high-res images into tiles processed independently.
- **Visual token count**: A typical 336x336 image produces 576 visual tokens (24x24 grid from ViT with patch size 14). A 1344x1344 image tiled into 4x4 = 16 tiles can produce 9,000+ tokens, consuming significant context.
- **Hallucination in vision**: Multimodal models frequently hallucinate visual details -- claiming to see objects that are not present. This is a major reliability challenge.
- **The frozen vs. unfrozen debate**: Some architectures keep the vision encoder frozen (preserving its pre-trained features) while training only the projection and LLM. Others fine-tune the vision encoder jointly, which improves alignment but risks catastrophic forgetting.
- **Interleaved training data**: Models trained on interleaved image-text documents (like web pages) develop stronger few-shot multimodal capabilities than those trained only on image-caption pairs.

## Common Misconceptions

- **"Multimodal models truly 'see' images like humans do"**: They process images as sequences of patch features. They lack the biological visual system's saccadic attention, depth perception from stereo vision, and constant visual grounding. Their "seeing" is statistical pattern matching on visual tokens.
- **"Adding vision to an LLM is straightforward"**: The alignment problem is deep. Simply concatenating visual features with text often produces models that ignore the visual input or hallucinate freely. Careful training recipes -- typically a two-stage process (alignment pre-training then instruction tuning) -- are required.
- **"Native multimodal is always better than bolted-on"**: Bolted-on approaches (frozen encoder + projection) are far more practical and can achieve competitive results. Native multimodal training requires enormous investment and may not justify the cost for all applications.
- **"These models understand video as motion"**: Most video models process sampled frames independently, lacking true temporal understanding. They recognize scenes and objects but often miss actions that require understanding motion dynamics between frames.

## Connections to Other Concepts

- **Contrastive Learning (CLIP, SigLIP)**: The foundation for most vision encoders used in multimodal models. Understanding contrastive loss is essential.
- **Cross-Attention**: The mechanism by which many models inject visual information into the language model, distinct from the self-attention used in standard text processing.
- **Context Window Extension**: Visual tokens consume context budget aggressively. Long-context techniques directly enable processing of higher-resolution images and longer videos.
- **RLHF and Alignment**: Multimodal models require specialized alignment -- including reducing visual hallucination -- using adapted RLHF pipelines that include image-based evaluation.
- **Compound AI Systems**: Multimodal models are themselves compound systems (encoder + projector + LLM), and they often serve as components in larger systems with tools, retrieval, and code execution.
- **Tokenization**: Understanding how continuous signals (pixels, audio waveforms) are converted into discrete or continuous token representations.

## Diagrams and Visualizations

*Recommended visual: Multimodal model architecture showing vision encoder connected to language model via projection layer — see [LLaVA Paper (arXiv:2304.08485)](https://arxiv.org/abs/2304.08485)*

*Recommended visual: Flamingo architecture showing cross-attention between visual features and language model — see [Alayrac et al. Flamingo Paper (arXiv:2204.14198)](https://arxiv.org/abs/2204.14198)*

## Further Reading

- **"Learning Transferable Visual Models From Natural Language Supervision" (Radford et al., 2021)**: The CLIP paper that established contrastive image-text pre-training as the foundation for modern multimodal AI.
- **"Visual Instruction Tuning" (Liu et al., 2023)**: The LLaVA paper demonstrating that a simple architecture (CLIP + linear projection + LLaMA) with good instruction-tuning data produces strong multimodal capabilities.
- **"Gemini: A Family of Highly Capable Multimodal Models" (Gemini Team, Google, 2024)**: Describes a natively multimodal architecture trained end-to-end on interleaved text, image, audio, and video data.
