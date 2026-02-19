# Quantization

**One-Line Summary**: Quantization reduces the numerical precision of a model's weights (and sometimes activations) from 16-bit floating point to 8-bit or 4-bit integers, shrinking memory footprint by 2-4x and accelerating inference, with surprisingly small losses in quality because neural networks are remarkably tolerant of reduced precision.

**Prerequisites**: Floating-point number representation (FP32, FP16, BF16), model parameters/weights, basic linear algebra (matrix multiplication), inference vs. training, GPU memory constraints.

## What Is Quantization?

Imagine you have a high-resolution photograph stored at 24 bits per pixel. Converting it to 8 bits per pixel reduces the file size by 3x, and for most purposes the image still looks fine -- you lose some subtle gradients, but the content is preserved. Quantization does the same thing to neural network weights: it reduces the precision of each number, trading a small amount of accuracy for dramatic savings in memory and computation.

*Recommended visual: Comparison of FP32, FP16, INT8, and INT4 precision formats showing bit layout and representable ranges — see [Hugging Face Quantization Guide](https://huggingface.co/docs/optimum/concept_guides/quantization)*


A 70B parameter model in FP16 requires about 140 GB of memory -- too large for a single GPU. Quantized to 4-bit integers, that same model fits in about 35 GB, comfortably fitting on a single 48 GB GPU. This is not just a convenience; it is often the difference between a model being deployable or not.

## How It Works


*Recommended visual: GPTQ vs AWQ vs GGUF quantization quality comparison across model sizes — see [Hugging Face Blog – Overview of Quantization](https://huggingface.co/blog/overview-quantization-transformers)*

### The Basics of Number Representation

- **FP32** (32-bit float): 1 sign bit, 8 exponent bits, 23 mantissa bits. High precision, 4 bytes per parameter.
- **FP16** (16-bit float): 1 sign bit, 5 exponent bits, 10 mantissa bits. Standard for training/inference, 2 bytes per parameter.
- **BF16** (bfloat16): 1 sign bit, 8 exponent bits, 7 mantissa bits. Same range as FP32, less precision. Preferred for training.
- **INT8** (8-bit integer): 256 discrete values. 1 byte per parameter.
- **INT4** (4-bit integer): 16 discrete values. 0.5 bytes per parameter.

### Quantization Formula

The core operation maps a continuous floating-point range to discrete integer levels:

$$x_{\text{quant}} = \text{round}\left(\frac{x - z}{s}\right), \quad x_{\text{dequant}} = x_{\text{quant}} \times s + z$$

where s is the **scale factor** and z is the **zero point**. The scale maps the floating-point range to the integer range, and the zero point handles asymmetric distributions.

For symmetric quantization (zero point = 0):

$$s = \frac{\max(|x|)}{2^{b-1} - 1}$$

where b is the target bit width. This maps the maximum absolute value to the largest representable integer.

### Post-Training Quantization (PTQ)

PTQ quantizes a model *after* it has been fully trained. No additional training is needed.

1. Take the pre-trained FP16 model.
2. Analyze the weight distributions (and optionally activation distributions using calibration data).
3. Compute scale factors and zero points for each layer or group of weights.
4. Convert weights to the target precision.

PTQ is fast and simple but can degrade quality, especially at very low bit widths (4-bit), because the quantization parameters must approximate the original distribution without any opportunity for the model to adapt.

### Quantization-Aware Training (QAT)

QAT simulates quantization during training so the model learns to be robust to reduced precision:

1. Insert "fake quantization" nodes that quantize and immediately dequantize values during the forward pass.
2. The model experiences quantization noise during training and adjusts its weights accordingly.
3. After training, apply real quantization.

QAT produces higher quality at low bit widths but requires significant compute (essentially fine-tuning the model). It is used when quality at 4-bit or lower is critical.

### GPTQ (GPU-Optimized Post-Training Quantization)

GPTQ (Frantar et al., 2023) is a one-shot weight quantization method optimized for LLMs:

- Quantizes weights one layer at a time, using a small calibration dataset (128-256 samples).
- Uses **Hessian information** (second-order optimization) to determine which weights are most sensitive and should be quantized more carefully.
- Processes weights in a specific order, using the Optimal Brain Quantizer (OBQ) framework, updating remaining weights to compensate for quantization error.
- Produces INT4 models that run efficiently on GPUs with minimal quality loss.

GPTQ is the go-to method for GPU-based INT4 quantization.

### AWQ (Activation-Aware Weight Quantization)

AWQ (Lin et al., 2024) observes that a small fraction of weights (about 1%) are disproportionately important because they correspond to large activation values:

- Identifies "salient" weight channels by examining activation magnitudes on calibration data.
- Applies per-channel scaling to protect these salient weights before quantization.
- Does not keep any weights in higher precision -- instead, the scaling makes the important weights easier to quantize accurately.

AWQ often achieves slightly better quality than GPTQ and is faster to apply.

### GGUF (CPU/Apple Silicon Optimized)

GGUF (used by llama.cpp and Ollama) is a file format and quantization approach designed for CPU and Apple Silicon inference:

- Supports a wide range of quantization levels: Q2_K, Q3_K, Q4_K, Q5_K, Q6_K, Q8_0, and more.
- Uses "k-quant" methods that mix different bit widths across layers, allocating more bits to sensitive layers.
- Optimized for CPU inference with ARM NEON and AVX instructions, and for Apple Silicon unified memory.
- Enables running large models on consumer hardware (MacBooks, gaming PCs) without a datacenter GPU.

## Why It Matters

Quantization is the single most impactful technique for making large models accessible. It directly enables:

- **Running 70B models on a single GPU** instead of requiring 2-4 GPUs.
- **Local inference on consumer hardware** via GGUF/llama.cpp.
- **Reduced serving costs** by fitting more concurrent requests in GPU memory.
- **Lower latency** because less data needs to be transferred from GPU memory (inference is memory-bandwidth-bound).

The memory bandwidth savings are particularly important: a 4-bit model reads 4x less data per forward pass, directly translating to faster token generation during the decode phase.

## Key Technical Details

- **Quality at different bit widths**: 8-bit is nearly lossless (< 0.1% perplexity increase). 4-bit is very good (0.5-2% perplexity increase). 3-bit shows noticeable degradation. 2-bit is generally impractical for most tasks.
- **Group quantization**: Instead of one scale factor per entire tensor, modern methods use one scale factor per group of 32-128 weights, significantly improving accuracy.
- **Weight-only vs. weight-and-activation quantization**: Most LLM quantization is weight-only (activations remain in FP16 during computation). Full INT8 weight+activation quantization (W8A8) is used in some high-throughput scenarios.
- **FP8**: A newer format supported by NVIDIA Hopper and Blackwell GPUs, offering a middle ground between FP16 and INT8 with the dynamic range of floats and the compactness of 8 bits.
- **Quantization is not compression in the information-theoretic sense**: The model is not "losing information" uniformly. Neural networks have substantial redundancy, and quantization exploits this.

## Common Misconceptions

- **"Quantized models are fundamentally worse."** At 8-bit and even 4-bit, quantized models perform remarkably close to their full-precision counterparts on most benchmarks. The quality difference is often smaller than the variation between different prompting strategies.
- **"Quantization is only for inference."** While most quantization is applied for inference, QAT happens during training, and there is active research on training in lower precision (FP8 training on Hopper GPUs).
- **"All 4-bit quantizations are equal."** GPTQ, AWQ, and GGUF k-quants at 4-bit can differ significantly in quality depending on the model architecture and calibration data. Method choice matters.
- **"You should always use the lowest bit width possible."** The right choice depends on your quality requirements and hardware. If you have enough memory for 8-bit, it is usually preferable to 4-bit. Only go lower when constrained.
- **"Quantization speeds up computation directly."** For weight-only quantization, the speedup comes from reduced memory bandwidth usage, not from faster arithmetic. The actual matrix multiplications often dequantize back to FP16 before computing.

## Connections to Other Concepts

- **KV Cache**: KV cache can also be quantized (KV cache quantization), which is orthogonal to weight quantization and addresses the memory bottleneck during long-context generation.
- **Knowledge Distillation**: Distillation creates a smaller model with fewer parameters; quantization keeps the same parameters but represents them with fewer bits. They can be combined for maximum compression.
- **Model Serving Frameworks**: vLLM supports AWQ and GPTQ; TensorRT-LLM supports FP8 and INT4; Ollama uses GGUF. Framework choice often determines quantization method.
- **Speculative Decoding**: If the target model is already quantized, the speedup from speculative decoding may be smaller (the target is already fast). However, using a tiny quantized draft model can still provide benefits.
- **Flash Attention**: Flash Attention operates on FP16/BF16 activations regardless of weight quantization, so the two optimizations are fully complementary.

## Further Reading

1. **"GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers"** (Frantar et al., 2023) -- The foundational paper for GPU-optimized LLM quantization using Hessian-based methods.
2. **"AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration"** (Lin et al., 2024) -- Introduces the insight that protecting activation-salient weights dramatically improves quantization quality.
3. **"The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits"** (Ma et al., 2024) -- A provocative paper on extreme quantization (ternary weights: -1, 0, +1), pushing the boundaries of how far precision can be reduced.
