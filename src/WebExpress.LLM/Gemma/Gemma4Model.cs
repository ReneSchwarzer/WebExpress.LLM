using System;
using WebExpress.LLM.Model;
using WebExpress.LLM.SafeTensors;
using WebExpress.LLM.Tensor;

namespace WebExpress.LLM.Gemma;

/// <summary>
/// Represents the complete Gemma-4 transformer model, orchestrating the forward pass
/// through all transformer layers from token embedding to vocabulary logits.
/// </summary>
/// <remarks>
/// The model consists of:
/// 1. Token embedding lookup with scaling
/// 2. 35 transformer layers, each with:
///    - RMS normalization
///    - Multi-head attention (sliding window or full)
///    - Residual connection
///    - Post-attention RMS normalization
///    - Gated feed-forward network
///    - Residual connection
/// 3. Final RMS normalization
/// 4. Linear projection to vocabulary logits
/// </remarks>
public sealed class Gemma4Model
{
    private readonly ModelConfiguration _config;
    private readonly ISafeTensorLoader _loader;
    private readonly KvCache _kvCache;

    /// <summary>
    /// Initializes a new Gemma4Model with the given configuration and weight loader.
    /// </summary>
    /// <param name="config">The model configuration.</param>
    /// <param name="loader">The SafeTensor loader providing access to model weights.</param>
    public Gemma4Model(ModelConfiguration config, ISafeTensorLoader loader)
    {
        _config = config ?? throw new ArgumentNullException(nameof(config));
        _loader = loader ?? throw new ArgumentNullException(nameof(loader));
        _kvCache = new KvCache();
    }

    /// <summary>
    /// Gets the KV cache for inspection or reset.
    /// </summary>
    public KvCache Cache => _kvCache;

    /// <summary>
    /// Performs a complete forward pass through the model, returning logits for
    /// the last token position.
    /// </summary>
    /// <remarks>
    /// Forward pass steps for Gemma 4:
    /// - Tokenize input text
    /// - Lookup embeddings for token IDs and scale by sqrt(hidden_size)
    /// - For each transformer layer (0..N-1):
    ///   - Apply RMS normalization to the input
    ///   - Multi-head attention with RoPE, sliding/global attention, and optional KV cache
    ///   - Add residual connection
    ///   - Apply RMS normalization after attention
    ///   - Feed-forward network (Dense or MoE/Gated, e.g. with GeLU activation)
    ///   - Add residual connection
    /// - Final RMS normalization
    /// - Project to vocabulary logits via matrix multiplication
    /// - (Optional) Apply logit scaling, softmax, and/or sampling as configured
    /// </remarks>
    /// <param name="tokenIds">The input token IDs.</param>
    /// <returns>An array of logit values, one per vocabulary entry.</returns>
    public float[] Forward(int[] tokenIds)
    {
        //System.Console.WriteLine($"Gemma4Model.Forward");

        ArgumentNullException.ThrowIfNull(tokenIds);

        if (tokenIds.Length == 0)
        {
            throw new ArgumentException("Token IDs must not be empty.", nameof(tokenIds));
        }

        var hiddenSize = _config.HiddenSize;
        var numLayers = _config.NumberOfLayers;
        var numQueryHeads = _config.NumberOfAttentionHeads;
        var numKvHeads = _config.NumberOfKeyValueHeads;
        var headDim = _config.HeadDimension;
        var rmsEps = _config.RmsNormEpsilon;

        // 1. Token embedding lookup
        var embedWeight = _loader.LoadTensor("model.language_model.embed_tokens.weight");
        var hidden = TensorOperations.EmbeddingLookup(embedWeight, tokenIds);

        // Scale embeddings by sqrt(hidden_size) as per Gemma convention
        hidden = hidden * MathF.Sqrt(hiddenSize);

        // 2. Process through transformer layers
        for (var layer = 0; layer < numLayers; layer++)
        {
            hidden = TransformerLayer(hidden, layer, numQueryHeads, numKvHeads, headDim, rmsEps);
        }

        // 3. Final RMS normalization
        var finalNormWeight = _loader.LoadTensor("model.language_model.norm.weight");
        hidden = TensorOperations.RmsNorm(hidden, finalNormWeight, rmsEps);

        // 4. Project to vocabulary logits
        // Get the last position's hidden state
        var lastHidden = hidden.GetLastRow();
        var lastHidden2D = lastHidden.Reshape(1, hiddenSize);

        Tensor.Tensor logits2D;

        // Check if word embeddings are tied (shared between embedding and output)
        if (_config.TieWordEmbeddings)
        {
            logits2D = TensorOperations.MatMul(lastHidden2D, embedWeight.Transpose());
        }
        else
        {
            var lmHeadWeight = _loader.LoadTensor("lm_head.weight");
            logits2D = TensorOperations.MatMul(lastHidden2D, lmHeadWeight.Transpose());
        }

        // 5. Apply logit softcapping if configured
        var softcapping = _config.TextConfig?.FinalLogitSoftcapping ?? 0;

        if (softcapping > 0)
        {
            logits2D = TensorOperations.Tanh(logits2D / softcapping) * softcapping;
        }

        return logits2D.GetRow(0).ToArray();
    }

    /// <summary>
    /// Processes a single transformer layer.
    /// </summary>
    /// <remarks>
    /// Performs all operations for a single transformer layer as used in Gemma 4.
    /// Steps:
    /// - Applies RMS normalization to the input hidden state.
    /// - Computes multi-head attention (with rotary embeddings, attention type, and key-value sharing as configured).
    /// - Adds the attention output via a residual connection.
    /// - Applies RMS normalization after the attention block.
    /// - Runs the feed-forward sublayer (e.g., gated or MoE variant as configured).
    /// - Adds the feed-forward output via a second residual connection.
    /// Layer-specific weights and attention settings are loaded dynamically for each layer and configuration.
    /// </remarks>
    /// <param name="hidden">The input hidden state tensor for this layer.</param>
    /// <param name="layerIndex">The index of the transformer layer (0-based).</param>
    /// <param name="numQueryHeads">The number of query heads for multi-head attention.</param>
    /// <param name="numKvHeads">The number of key-value heads for multi-head attention.</param>
    /// <param name="headDim">The dimension of each attention head.</param>
    /// <param name="rmsEps">The epsilon value for RMS normalization.</param>
    private Tensor.Tensor TransformerLayer(
        Tensor.Tensor hidden, int layerIndex,
        int numQueryHeads, int numKvHeads, int headDim, float rmsEps)
    {
        var prefix = $"model.language_model.layers.{layerIndex}";
        //System.Console.WriteLine($"Gemma4Model.TransformerLayer {prefix}");

        // Determine attention type for this layer
        var layerTypes = _config.TextConfig?.LayerTypes;
        var isFullAttention = false;

        if (layerTypes != null && layerIndex < layerTypes.Count)
        {
            isFullAttention = layerTypes[layerIndex] == "full_attention";
        }

        // Determine head dimension based on attention type
        var effectiveHeadDim = headDim;

        if (isFullAttention && _config.TextConfig?.GlobalHeadDimension > 0)
        {
            effectiveHeadDim = _config.TextConfig.GlobalHeadDimension;
        }

        // Create RoPE for this layer type
        var ropeParams = _config.TextConfig?.RopeParameters;
        var ropeEntry = isFullAttention ? ropeParams?.FullAttention : ropeParams?.SlidingAttention;
        var theta = ropeEntry?.RopeTheta ?? 10000.0f;
        var partialFactor = ropeEntry?.PartialRotaryFactor ?? 1.0f;
        var rope = new RotaryEmbedding(theta, partialFactor);

        // 1. Input RMS normalization
        var inputNormWeight = _loader.LoadTensor($"{prefix}.input_layernorm.weight");
        var normalized = TensorOperations.RmsNorm(hidden, inputNormWeight, rmsEps);

        // 2. Multi-head attention
        var qWeight = _loader.LoadTensor($"{prefix}.self_attn.q_proj.weight");
        var kWeight = _loader.LoadTensor($"{prefix}.self_attn.k_proj.weight");

        // When attention_k_eq_v is true, K and V projections share the same weight
        // and no separate v_proj.weight tensor exists in the model files.
        var keyEqualsValue = _config.TextConfig?.AttentionKeyEqualsValue ?? false;
        var vWeight = keyEqualsValue
            ? kWeight
            : _loader.LoadTensor($"{prefix}.self_attn.v_proj.weight");

        var oWeight = _loader.LoadTensor($"{prefix}.self_attn.o_proj.weight");

        var slidingWindow = _config.TextConfig?.SlidingWindow ?? 512;

        var attention = new MultiHeadAttention(
            numQueryHeads, numKvHeads, effectiveHeadDim,
            isFullAttention, slidingWindow, rope);

        var attended = attention.Forward(
            normalized, qWeight, kWeight, vWeight, oWeight,
            _kvCache, layerIndex);

        // 3. Residual connection
        var residual1 = hidden + attended;

        // 4. Post-attention RMS normalization
        var postNormWeight = _loader.LoadTensor($"{prefix}.post_attention_layernorm.weight");
        var normalized2 = TensorOperations.RmsNorm(residual1, postNormWeight, rmsEps);

        // 5. Feed-forward network
        var gateWeight = _loader.LoadTensor($"{prefix}.mlp.gate_proj.weight");
        var upWeight = _loader.LoadTensor($"{prefix}.mlp.up_proj.weight");
        var downWeight = _loader.LoadTensor($"{prefix}.mlp.down_proj.weight");
        var ffOutput = FeedForward.Forward(normalized2, gateWeight, upWeight, downWeight);

        // 6. Residual connection
        return residual1 + ffOutput;
    }

    /// <summary>
    /// Resets the KV cache, typically called at the start of a new generation session.
    /// </summary>
    public void ResetCache()
    {
        _kvCache.Clear();
    }
}
