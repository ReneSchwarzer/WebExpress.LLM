# Gemma-4 Integration Guide

## Current State

The `TransformerInferenceEngine` is currently a **placeholder implementation** that demonstrates the architecture and provides async streaming capabilities, but does not perform actual Gemma-4 transformer inference.

### What Works Now
- ✅ Async streaming token generation with `IAsyncEnumerable<int>`
- ✅ Model configuration loading from `config.json`
- ✅ Model weights loading (SafeTensors, PyTorch bin files)
- ✅ Sampling strategies (Greedy, Top-K, Top-P)
- ✅ Chat session management with conversation history
- ✅ ByteTokenizer for UTF-8 encoding/decoding
- ✅ Memory-mapped file support for large model weights (>2GB)

### What Needs Implementation
The `ForwardPass` method in `TransformerInferenceEngine` currently generates placeholder logits. For proper Gemma-4 inference, the following components must be implemented:

## Required Components for Full Gemma-4 Support

### 1. Tensor Operations Library
**Options:**
- **TorchSharp** (recommended): C# bindings for PyTorch, GPU support
- **NumSharp**: Pure C# NumPy-like library
- **ONNX Runtime**: If converting Gemma-4 to ONNX format
- **ML.NET**: Microsoft's ML framework

**Recommendation**: Use TorchSharp for best compatibility with PyTorch-based models.

```csharp
// Example dependency
<PackageReference Include="TorchSharp-cpu" Version="0.100.6" />
// or for GPU support:
<PackageReference Include="TorchSharp-cuda-windows" Version="0.100.6" />
```

### 2. SafeTensors Parser
Parse the SafeTensors weight format to extract model parameters.

**Key weights to extract:**
- `model.embed_tokens.weight` - Token embeddings (vocab_size × hidden_size)
- `model.layers.{i}.self_attn.q_proj.weight` - Query projection
- `model.layers.{i}.self_attn.k_proj.weight` - Key projection
- `model.layers.{i}.self_attn.v_proj.weight` - Value projection
- `model.layers.{i}.self_attn.o_proj.weight` - Output projection
- `model.layers.{i}.mlp.gate_proj.weight` - MLP gate projection
- `model.layers.{i}.mlp.up_proj.weight` - MLP up projection
- `model.layers.{i}.mlp.down_proj.weight` - MLP down projection
- `model.layers.{i}.input_layernorm.weight` - RMS norm weights
- `model.layers.{i}.post_attention_layernorm.weight` - Post-attention norm
- `lm_head.weight` - Final output projection to vocabulary

### 3. Embedding Layer Implementation

```csharp
private Tensor GetEmbeddings(IReadOnlyList<int> tokenIds)
{
    // Load token embeddings from weights
    var embeddings = _weights["model.embed_tokens.weight"];

    // Look up embeddings for input tokens
    var inputEmbeddings = embeddings[tokenIds];

    // Apply RoPE positional embeddings based on layer type
    return ApplyPositionalEmbeddings(inputEmbeddings);
}
```

### 4. Transformer Layer Implementation

For each of the 35 layers in Gemma-4:

```csharp
private Tensor TransformerLayer(Tensor input, int layerIndex)
{
    // 1. Input RMS normalization
    var normalized = RMSNorm(input, layerIndex, "input_layernorm");

    // 2. Multi-head attention (sliding or full based on layer_types config)
    var attended = MultiHeadAttention(normalized, layerIndex);

    // 3. Residual connection
    var residual1 = input + attended;

    // 4. Post-attention normalization
    var normalized2 = RMSNorm(residual1, layerIndex, "post_attention_layernorm");

    // 5. Feed-forward network with gated activation
    var ffOutput = FeedForward(normalized2, layerIndex);

    // 6. Residual connection
    return residual1 + ffOutput;
}
```

### 5. Attention Mechanism

Gemma-4 uses two attention patterns:
- **Sliding Window Attention** (most layers): 512-token window
- **Full Attention** (layers 4, 9, 14, 19, 24, 29, 34): Attend to all tokens

```csharp
private Tensor MultiHeadAttention(Tensor input, int layerIndex)
{
    var layerType = _config.TextConfig.LayerTypes[layerIndex];
    var isFullAttention = layerType == "full_attention";

    // Project to Q, K, V
    var Q = MatMul(input, _weights[$"model.layers.{layerIndex}.self_attn.q_proj.weight"]);
    var K = MatMul(input, _weights[$"model.layers.{layerIndex}.self_attn.k_proj.weight"]);
    var V = MatMul(input, _weights[$"model.layers.{layerIndex}.self_attn.v_proj.weight"]);

    // Apply RoPE to Q and K
    Q = ApplyRoPE(Q, layerType);
    K = ApplyRoPE(K, layerType);

    // Compute attention scores
    var scores = MatMul(Q, K.Transpose()) / Math.Sqrt(headDim);

    // Apply sliding window mask if needed
    if (!isFullAttention)
    {
        scores = ApplySlidingWindowMask(scores, windowSize: 512);
    }

    // Apply attention and project output
    var attention = Softmax(scores);
    var output = MatMul(attention, V);
    return MatMul(output, _weights[$"model.layers.{layerIndex}.self_attn.o_proj.weight"]);
}
```

### 6. RoPE (Rotary Position Embedding)

Gemma-4 uses different RoPE configurations for sliding vs full attention:

```csharp
private Tensor ApplyRoPE(Tensor tensor, string layerType)
{
    var ropeConfig = layerType == "full_attention"
        ? _config.TextConfig.RopeParameters.FullAttention
        : _config.TextConfig.RopeParameters.SlidingAttention;

    var theta = ropeConfig.RopeTheta;
    var ropeType = ropeConfig.RopeType;

    // Implement RoPE based on type (default or proportional)
    // Apply partial rotation if specified (full_attention uses 0.25)
    return ApplyRotaryEmbedding(tensor, theta, ropeType);
}
```

### 7. RMS Normalization

```csharp
private Tensor RMSNorm(Tensor input, int layerIndex, string normType)
{
    var epsilon = _config.TextConfig.RmsNormEps; // 1e-6
    var weight = _weights[$"model.layers.{layerIndex}.{normType}.weight"];

    // RMS norm: x * weight / sqrt(mean(x^2) + epsilon)
    var variance = input.Pow(2).Mean(dim: -1, keepdim: true);
    return input * weight / (variance + epsilon).Sqrt();
}
```

### 8. Feed-Forward Network

Gemma-4 uses gated activation (GELU):

```csharp
private Tensor FeedForward(Tensor input, int layerIndex)
{
    var gateProj = _weights[$"model.layers.{layerIndex}.mlp.gate_proj.weight"];
    var upProj = _weights[$"model.layers.{layerIndex}.mlp.up_proj.weight"];
    var downProj = _weights[$"model.layers.{layerIndex}.mlp.down_proj.weight"];

    // Gated activation: down(GELU(gate(x)) * up(x))
    var gate = MatMul(input, gateProj);
    var gated = GELU(gate);
    var up = MatMul(input, upProj);
    var combined = gated * up;
    return MatMul(combined, downProj);
}
```

### 9. Complete Forward Pass

```csharp
private float[] ForwardPass(IReadOnlyList<int> tokens)
{
    // 1. Get token embeddings
    var hidden = GetEmbeddings(tokens);

    // 2. Process through all 35 transformer layers
    for (int i = 0; i < _config.TextConfig.NumHiddenLayers; i++)
    {
        hidden = TransformerLayer(hidden, i);
    }

    // 3. Final normalization
    hidden = FinalRMSNorm(hidden);

    // 4. Project to vocabulary logits
    var lmHead = _weights["lm_head.weight"];
    var logits = MatMul(hidden, lmHead);

    // 5. Apply logit softcapping if configured
    if (_config.TextConfig.FinalLogitSoftcapping.HasValue)
    {
        var cap = _config.TextConfig.FinalLogitSoftcapping.Value;
        logits = logits.Tanh() * cap;
    }

    // 6. Return logits for last position (autoregressive generation)
    return logits[-1].ToArray();
}
```

### 10. KV Cache for Efficiency

For autoregressive generation, cache key/value projections:

```csharp
private class KVCache
{
    private Dictionary<int, (Tensor Keys, Tensor Values)> _cache = new();

    public void Update(int layerIndex, Tensor keys, Tensor values)
    {
        if (_cache.ContainsKey(layerIndex))
        {
            // Concatenate with existing cache
            var (existingKeys, existingValues) = _cache[layerIndex];
            _cache[layerIndex] = (
                Tensor.Cat(existingKeys, keys, dim: 1),
                Tensor.Cat(existingValues, values, dim: 1)
            );
        }
        else
        {
            _cache[layerIndex] = (keys, values);
        }
    }

    public (Tensor Keys, Tensor Values) Get(int layerIndex) => _cache[layerIndex];
}
```

## Implementation Roadmap

1. **Phase 1**: Add TorchSharp dependency and basic tensor operations
2. **Phase 2**: Implement SafeTensors parser to load weights
3. **Phase 3**: Implement embedding layer with RoPE
4. **Phase 4**: Implement RMS normalization
5. **Phase 5**: Implement attention mechanism (both sliding and full)
6. **Phase 6**: Implement feed-forward network
7. **Phase 7**: Connect all layers in ForwardPass
8. **Phase 8**: Add KV cache for efficient generation
9. **Phase 9**: Optimize and test with actual Gemma-4 weights
10. **Phase 10**: Add GPU acceleration support

## Testing Strategy

1. Start with a tiny test model (few layers, small hidden size)
2. Verify each component independently:
   - Embeddings match expected shape
   - RMS norm produces correct statistics
   - Attention outputs are in valid range
   - Forward pass completes without errors
3. Compare outputs with reference implementation (e.g., HuggingFace Transformers in Python)
4. Gradually scale up to full Gemma-4 model

## Performance Considerations

- **Memory**: Gemma-4-E2B-it requires ~8-16GB RAM for weights
- **Speed**: CPU inference will be slow (seconds per token). GPU recommended.
- **Precision**: Use bfloat16 or float16 for weights to save memory
- **Batching**: Consider batch inference for multiple requests

## References

- [Gemma-4 Model Card](https://huggingface.co/google/gemma-4-E2B-it)
- [SafeTensors Format](https://github.com/huggingface/safetensors)
- [TorchSharp Documentation](https://github.com/dotnet/TorchSharp)
- [RoPE: Rotary Position Embedding](https://arxiv.org/abs/2104.09864)
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
