# Gemma-4 Integration Guide

## Current State

The `TransformerInferenceEngine` is a **fully functional implementation** of the Gemma-4 transformer architecture. It supports both high-performance sharded model loading and single-file SafeTensors, performing a complete forward pass through all 35 transformer layers.

## Process Flow

The following process flow outlines each step performed by a Gemma 4 large language model when generating a response to an input prompt.
It describes how user input is transformed into model-understandable tokens, processed through successive layers of neural computation, and finally decoded back into human-readable output.
Understanding this pipeline is essential for anyone integrating, optimizing, or analyzing inference behavior in Gemma-based systems.
Each step ensures that the model makes maximum use of its architecture to provide coherent, context-aware responses.

1. **User Input Reception:** The user provides an input string—such as a question, command, or conversation prompt—intended for the model.

2. **Tokenization:** The input string is transformed into a sequence of numerical token IDs using the model's specific tokenizer. This ensures text is in the discrete form the neural network expects.
Additional formatting or system tokens may be applied for chat, role-play, or instruction-following.

3. **Embedding Lookup:** The list of token IDs is mapped to vectors using the embedding matrix ([vocab_size, hidden_size]) provided by the model. Each ID selects an embedding vector. This results in an embedding tensor of shape `[seq_len, hidden_size]`.

4. **Embedding Scaling:** The embeddings are scaled—typically by multiplying with sqrt(hidden_size)—which stabilizes the input variance as described in the model’s paper.

5. **Passing Through Transformer Layers:** For each transformer layer (Gemma 4 use 30 layers, depending on the variant) the following sub-steps are repeated sequentially:

    a. **RMS Normalization:** The current hidden tensor is normalized using RMSNorm (Root Mean Square Layer Norm), improving training and inference stability without the need for learned bias parameters.

    b. **Multi-Head Self-Attention:** The hidden tensor is projected into queries, keys, and values via learned linear weights. Rotary position embeddings (RoPE) are applied to inject positional information. Depending on the layer, attention may be "sliding window" (local neighborhoods) or "full" (global tokens). The attention operation computes context-aware representations by mixing token information, potentially using a key-value cache for efficient long-sequence generation.
     
    c. **Residual Connection:** The output of the self-attention is added back to the input of the block, preserving direct information flow as in all transformers.

    d. **Second RMS Normalization:** The tensor is normalized again, prior to the feed-forward operation.

    e. **Feed-Forward Neural Network (FFN / MoE / Gated FFN):** Each position undergoes a two- or three-layer neural network (often with gating/sparse Mixture-of-Experts in larger models), enabling richer nonlinear transformations.

    f. **Second Residual Connection:** The output of the feed-forward block is again added to the incoming tensor (post-attention), completing the transformer layer.

6. **Final RMS Normalization:** Once all layers have been processed, a final RMSNorm is applied to the tensor, ensuring model output consistency.

7. **Projection to Vocabulary Logits:** The hidden state for the last processed token (position) is multiplied (dot product) with the transpose of the embedding matrix (or a dedicated output matrix), converting [hidden_size] to [vocab_size]—one score per possible next token.

8. **Logit Processing and Sampling:** The logits (raw scores) may optionally be transformed by: 

    Applying temperature scaling Softmax or logit-capping (e.g., tanh with temperature, as in some Gemma variants) Then, the model samples the next token: Greedy (argmax): always picks the highest-scoring token. Stochastic (sampling): samples according to probability, possibly using nucleus/top-p or similar strategies. This determines the next token’s index.

9. **Autoregressive Decoding Loop (for generating multiple tokens);** The predicted token is appended to the context. The process from step 5 onward is repeated until the desired number of tokens is generated or a stop condition (such as EOS token) is met. Optimally, KV-caches and attention masks are managed so only new parts of the sequence are processed repeatedly.

10. **Detokenization:** The generated sequence of token IDs is mapped back to text using the inverse of the tokenizer, reconstructing human-readable output.

11. **Delivering the Model Output:** The generated text (possibly streamed token-by-token or as one complete response) is presented to the user as the model's reply.

### What Works
- **Full Transformer Inference**: Complete forward pass with attention, normalization, and FFN.
- **Custom Tensor Library**: Efficient, native .NET tensor implementation in `WebExpress.LLM.Tensor`.
- **Async Streaming**: Real-time token generation with `IAsyncEnumerable<int>`.
- **Sharded Weight Loading**: Efficiently handle multi-file models (e.g., `model-00001-of-00002.safetensors`).
- **Memory-Mapped Files**: Native support for large model weights (>2GB) using shared memory.
- **KV Cache**: Efficient autoregressive generation by caching previous keys and values.
- **Sampling Strategies**: Greedy, Top-K, and Top-P sampling implemented.
- **ByteTokenizer**: Full UTF-8 support for robust encoding/decoding.

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
Each layer (as defined in `Gemma4Model.TransformerLayer`) performs:

1.  **Input RMS Normalization**: Normalizes inputs using the `input_layernorm` weights.
2.  **Multi-Head Attention**:
    - Supports both **Sliding Window Attention** (default) and **Full Attention** (global layers).
    - Implements **GQA** (Grouped Query Attention) based on head configuration.
    - Uses a distinct KV-head count for full-attention layers (`num_global_key_value_heads`).
    - Optional per-head **QK-Norm** (`self_attn.q_norm`, `self_attn.k_norm`) before RoPE.
    - Optional **attention-logits soft cap** (`attn_logit_softcapping`) applied pre-softmax.
    - Handles `attention_key_equals_value` for specific model variants.
3.  **Skip-Scaled Residual Connection**: Adds the attention output (multiplied by the per-layer `layer_scalar`) to the input.
4.  **Post-Attention Normalization**: Second RMS norm before the feed-forward stage.
5.  **Feed-Forward Stage**:
    - **MoE variants** (`enable_moe_block=true`, e.g. 26B_A4B): a Mixture-of-Experts branch and a dense shared branch (mlp2) run in parallel, each with its own pre- and post-RMSNorm; the outputs are summed and a combined post-FFW RMSNorm is applied before the residual.
    - **Non-MoE variants**: a single gated feed-forward network (`gate_proj`/`up_proj`/`down_proj`).
6.  **Skip-Scaled Final Residual Connection**: Adds the feed-forward output (multiplied by `layer_scalar`) back to the residual.

### 4. Rotary Positional Embeddings (RoPE)
Implemented in the `RotaryEmbedding` class, supporting:
- Configurable `theta` values.
- **Partial Rotary Factors** (e.g., 0.25 for full attention layers).
- Smooth frequency calculation for both sliding and full attention contexts.

### 5. Efficient Generation (KV Cache)
The `KvCache` class manages the storage of key and value tensors across generation steps, drastically reducing the computational cost for long sequences by avoiding redundant processing of previous tokens.

## Deferred Features (Checked at Load Time)

`Gemma4Model.Forward` refuses to run a checkpoint that requests any of the
following — guarded to prevent silent mis-inference:

- `hidden_size_per_layer_input > 0` — per-layer input (PLE) projections. Not
  used by 26B_A4B. Reference implementation:
  `gemma/gm/nn/gemma4/layers.py` (`PerLayerInputProjection`).
- `num_kv_shared_layers > 0` — KV-cache sharing where trailing layers reuse
  K/V from an earlier layer. Not used by 26B_A4B.
- `use_double_wide_mlp == true` — double-wide dense MLP variant.

Each guard throws `NotSupportedException` on model invocation rather than
returning wrong tokens.

## Implementation Roadmap (Next Steps)

1.  **Phase 1**: SIMD optimization for `TensorOperations` to improve CPU performance.
2.  **Phase 2**: Optional GPU acceleration via Compute Shaders or DirectCompute.
3.  **Phase 3**: Quantization support (Q4_K, Q8_0) to reduce memory footprint.
4.  **Phase 4**: Multi-modal support (Vision/Audio) as per Gemma-4 specifications.
5.  **Phase 5**: Lift the deferred-feature guards listed above (PLE, KV-cache sharing, double-wide MLP).

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
