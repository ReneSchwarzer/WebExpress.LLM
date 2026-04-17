# Gemma-4 Integration Guide

## Current State

The `TransformerInferenceEngine` is now a **fully functional implementation** of the Gemma-4 transformer architecture. It supports both high-performance sharded model loading and single-file SafeTensors, performing a complete forward pass through all 35 transformer layers.

### What Works Now
- ✅ **Full Transformer Inference**: Complete forward pass with attention, normalization, and FFN.
- ✅ **Custom Tensor Library**: Efficient, native .NET tensor implementation in `WebExpress.LLM.Tensor`.
- ✅ **Async Streaming**: Real-time token generation with `IAsyncEnumerable<int>`.
- ✅ **Sharded Weight Loading**: Efficiently handle multi-file models (e.g., `model-00001-of-00002.safetensors`).
- ✅ **Memory-Mapped Files**: Native support for large model weights (>2GB) using shared memory.
- ✅ **KV Cache**: Efficient autoregressive generation by caching previous keys and values.
- ✅ **Sampling Strategies**: Greedy, Top-K, and Top-P sampling implemented.
- ✅ **ByteTokenizer**: Full UTF-8 support for robust encoding/decoding.

### Fallback Mechanism
If valid SafeTensors weights are not available or the model format is unsupported, the engine falls back to a **placeholder implementation** that generates readable English-biased text for development and integration testing.

## Implemented Gemma-4 Architecture

The model is implemented in pure C# using a custom tensor operations library, avoiding heavy external dependencies like TorchSharp.

### 1. Custom Tensor Library (`WebExpress.LLM.Tensor`)
Instead of using external libraries, the project uses a highly optimized `Tensor` class that supports:
- Multi-dimensional shapes and strides.
- Efficient memory access via `Span<float>`.
- Common operations: `MatMul`, `RmsNorm`, `EmbeddingLookup`, `Softmax`, `Tanh`, and element-wise arithmetic.

### 2. Weight Loading & Management
Weights are loaded via the `ISafeTensorLoader` interface, which supports:
- **Single File**: `SafeTensorLoader` for standalone `.safetensors` files.
- **Sharded**: `ShardedSafeTensorLoader` using a `model.safetensors.index.json` weight map.

### 3. Transformer Layer Implementation
Each of the 35 layers (as defined in `Gemma4Model.TransformerLayer`) performs:

1.  **Input RMS Normalization**: Normalizes inputs using the `input_layernorm` weights.
2.  **Multi-Head Attention**:
    - Supports both **Sliding Window Attention** (default) and **Full Attention** (global layers).
    - Implements **GQA** (Grouped Query Attention) based on head configuration.
    - Handles `attention_key_equals_value` for specific model variants.
3.  **Residual Connection**: Adds the attention output back to the input.
4.  **Post-Attention Normalization**: Second RMS norm before the MLP.
5.  **Gated Feed-Forward Network**: Implements `gate_proj`, `up_proj`, and `down_proj` with gated activation.
6.  **Final Residual Connection**: Produces the layer output.

### 4. Rotary Positional Embeddings (RoPE)
Implemented in the `RotaryEmbedding` class, supporting:
- Configurable `theta` values.
- **Partial Rotary Factors** (e.g., 0.25 for full attention layers).
- Smooth frequency calculation for both sliding and full attention contexts.

### 5. Efficient Generation (KV Cache)
The `KvCache` class manages the storage of key and value tensors across generation steps, drastically reducing the computational cost for long sequences by avoiding redundant processing of previous tokens.

## Implementation Roadmap (Next Steps)

1.  **Phase 1**: SIMD optimization for `TensorOperations` to improve CPU performance.
2.  **Phase 2**: Optional GPU acceleration via Compute Shaders or DirectCompute.
3.  **Phase 3**: Quantization support (Q4_K, Q8_0) to reduce memory footprint.
4.  **Phase 4**: Multi-modal support (Vision/Audio) as per Gemma-4 specifications.

## Testing Strategy

The implementation is verified through:
1.  **Unit Tests**: Individual components (`RotaryEmbedding`, `KvCache`, `TensorOperations`) are tested for mathematical correctness.
2.  **Component Tests**: `MultiHeadAttention` and `FeedForward` are tested against expected shapes and outputs.
3.  **Integration Tests**: `ModelLoader` and `SafeTensorIndex` ensure correct weight mapping for complex sharded models.

## References

- [Gemma-4 Model Card](https://huggingface.co/google/gemma-4-E2B-it)
- [SafeTensors Format](https://github.com/huggingface/safetensors)
- [RoPE: Rotary Position Embedding](https://arxiv.org/abs/2104.09864)
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
