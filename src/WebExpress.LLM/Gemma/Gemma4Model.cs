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
/// 1. Token embedding lookup with scaling by sqrt(hidden_size)
/// 2. N transformer layers, each with (matching Gemma-4 reference Block.__call__):
///    - Pre-attention RMS norm (input_layernorm)
///    - Multi-head attention (sliding window or full) with QK-Norm, value_norm
///    - Post-attention RMS norm applied to the attention output
///    - First residual: attn_output + input
///    - Feed-forward stage on the residual:
///        * Dense path: pre_feedforward_layernorm → gated MLP → post_feedforward_layernorm
///        * MoE path: pre/post norms around an MoE branch and a parallel dense shared
///          (mlp2) branch, summed and passed through a combined post-FFW norm.
///    - Second residual: ffw_output + attn_residual
///    - Multiplied once by the per-layer skip_scale (layer_scalar)
/// 3. Final RMS normalization
/// 4. Linear projection to vocabulary logits (tied with the embedding matrix when configured)
/// 5. Optional final logit soft-capping via tanh
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

        // Deferred-feature guards. 26B_A4B sets both to zero; other Gemma-4
        // variants may enable them and would silently produce wrong outputs
        // without the dedicated code paths.
        if (_config.TextConfig?.HiddenSizePerLayerInput > 0)
        {
            throw new NotSupportedException(
                "Per-layer input (PLE) projections are not yet supported. " +
                "See docs/GEMMA4_INTEGRATION.md for the deferred-feature list.");
        }

        if (_config.TextConfig?.NumberOfKvSharedLayers > 0)
        {
            throw new NotSupportedException(
                "KV-cache sharing across layers (num_kv_shared_layers > 0) is not yet supported. " +
                "See docs/GEMMA4_INTEGRATION.md for the deferred-feature list.");
        }

        if (_config.TextConfig?.UseDoubleWideMlp == true)
        {
            throw new NotSupportedException(
                "use_double_wide_mlp is not yet supported.");
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
    /// Mirrors the Gemma-4 reference block (<c>gemma/gm/nn/gemma4/_modules.Block</c>):
    /// <code>
    ///   x_norm        = pre_attention_norm(x)
    ///   attn          = attention(x_norm)
    ///   attn          = post_attention_norm(attn)        // norm on attn output
    ///   attn_residual = attn + x                          // first residual
    ///   ffw           = dense_or_moe_branch(attn_residual)
    ///   out           = ffw + attn_residual               // second residual
    ///   out           = out * skip_scale                  // single per-layer scale
    /// </code>
    /// The dense branch is <c>pre_ffw_norm → mlp → post_ffw_norm</c>; the MoE
    /// branch additionally adds a parallel dense shared (mlp2) branch and
    /// applies a final combined post-FFW norm. See <see cref="MoeAndSharedBranch"/>.
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

        // Determine attention type for this layer
        var layerTypes = _config.TextConfig?.LayerTypes;
        var isFullAttention = false;

        if (layerTypes != null && layerIndex < layerTypes.Count)
        {
            isFullAttention = layerTypes[layerIndex] == "full_attention";
        }

        // Full-attention layers may use a different number of KV heads
        // (e.g. gemma-4 26B_A4B: 8 sliding KV heads vs 2 global KV heads).
        var effectiveKvHeads = isFullAttention && _config.TextConfig?.NumberOfGlobalKeyValueHeads > 0
            ? _config.TextConfig.NumberOfGlobalKeyValueHeads
            : numKvHeads;

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

        // Per-layer skip scale ("layer_scalar" in the checkpoint, "skip_scale"
        // in the reference). Multiplied once at the end of the block. Optional:
        // if the tensor is absent we fall back to 1.0.
        var layerScalarTensor = _loader.TryLoadTensor($"{prefix}.layer_scalar");
        var skipScale = layerScalarTensor != null && layerScalarTensor.Length > 0
            ? layerScalarTensor[0]
            : 1.0f;

        // 1. Pre-attention RMS norm
        var inputNormWeight = _loader.LoadTensor($"{prefix}.input_layernorm.weight");
        var normalized = TensorOperations.RmsNorm(hidden, inputNormWeight, rmsEps);

        // 2. Multi-head attention (with optional QK-Norm)
        var qWeight = _loader.LoadTensor($"{prefix}.self_attn.q_proj.weight");
        var kWeight = _loader.LoadTensor($"{prefix}.self_attn.k_proj.weight");

        // When attention_k_eq_v is true, K and V projections share the same weight
        // and no separate v_proj.weight tensor exists in the model files.
        var keyEqualsValue = _config.TextConfig?.AttentionKeyEqualsValue ?? false;
        var vWeight = keyEqualsValue
            ? kWeight
            : _loader.LoadTensor($"{prefix}.self_attn.v_proj.weight");

        var oWeight = _loader.LoadTensor($"{prefix}.self_attn.o_proj.weight");

        var qNormWeight = _loader.TryLoadTensor($"{prefix}.self_attn.q_norm.weight");
        var kNormWeight = _loader.TryLoadTensor($"{prefix}.self_attn.k_norm.weight");

        var slidingWindow = _config.TextConfig?.SlidingWindow ?? 512;
        var attnSoftcap = _config.TextConfig?.AttentionLogitsSoftcapping ?? 0f;

        var attention = new MultiHeadAttention(
            numQueryHeads, effectiveKvHeads, effectiveHeadDim,
            isFullAttention, slidingWindow, rope, attnSoftcap);

        var attnOutput = attention.Forward(
            normalized, qWeight, kWeight, vWeight, oWeight,
            _kvCache, layerIndex,
            qNormWeight: qNormWeight,
            kNormWeight: kNormWeight,
            rmsNormEpsilon: rmsEps);

        // 3. Post-attention norm — applied to the attention output, not to the
        //    residual stream (matches `Block.__call__` in the reference).
        var postAttnNormWeight = _loader.LoadTensor($"{prefix}.post_attention_layernorm.weight");
        attnOutput = TensorOperations.RmsNorm(attnOutput, postAttnNormWeight, rmsEps);

        // 4. First residual: attn_output + input
        var attnResidual = attnOutput + hidden;

        // 5. Feed-forward stage on attnResidual.
        var enableMoe = _config.TextConfig?.EnableMoeBlock ?? false;
        Tensor.Tensor ffOutput;

        if (enableMoe)
        {
            ffOutput = MoeAndSharedBranch(attnResidual, prefix, rmsEps);
        }
        else
        {
            ffOutput = DenseFeedForward(attnResidual, prefix, rmsEps);
        }

        // 6. Second residual + single per-layer skip_scale.
        return (ffOutput + attnResidual) * skipScale;
    }

    /// <summary>
    /// Runs the standard (non-MoE) feed-forward branch:
    /// <c>pre_feedforward_layernorm → gated MLP → post_feedforward_layernorm</c>.
    /// </summary>
    /// <param name="attnResidual">Input activations after the first residual.</param>
    /// <param name="prefix">Layer prefix, e.g. "model.language_model.layers.3".</param>
    /// <param name="rmsEps">Epsilon used by all RMSNorm stages.</param>
    private Tensor.Tensor DenseFeedForward(Tensor.Tensor attnResidual, string prefix, float rmsEps)
    {
        var preFfwNorm = _loader.LoadTensor($"{prefix}.pre_feedforward_layernorm.weight");
        var postFfwNorm = _loader.TryLoadTensor($"{prefix}.post_feedforward_layernorm.weight");

        var ffwIn = TensorOperations.RmsNorm(attnResidual, preFfwNorm, rmsEps);

        var gateWeight = _loader.LoadTensor($"{prefix}.mlp.gate_proj.weight");
        var upWeight = _loader.LoadTensor($"{prefix}.mlp.up_proj.weight");
        var downWeight = _loader.LoadTensor($"{prefix}.mlp.down_proj.weight");
        var ffwOut = FeedForward.Forward(ffwIn, gateWeight, upWeight, downWeight);

        if (postFfwNorm != null)
        {
            ffwOut = TensorOperations.RmsNorm(ffwOut, postFfwNorm, rmsEps);
        }

        return ffwOut;
    }

    /// <summary>
    /// Runs the combined MoE + dense-shared (mlp2) feed-forward stage used in
    /// Gemma-4 MoE variants. Mirrors the reference's <c>_forward_moe</c>:
    /// <code>
    ///   dense_out = post_ffw2_norm(mlp2(pre_ffw2_norm(x)))
    ///   moe_out   = post_ffw1_norm(moe(pre_ffw_norm(x)))
    ///   out       = post_ffw_norm(dense_out + moe_out)
    /// </code>
    /// </summary>
    /// <param name="attnResidual">Input activations after the first residual (no extra norm applied).</param>
    /// <param name="prefix">Layer prefix, e.g. "model.language_model.layers.3".</param>
    /// <param name="rmsEps">Epsilon used by all RMSNorm stages.</param>
    private Tensor.Tensor MoeAndSharedBranch(Tensor.Tensor attnResidual, string prefix, float rmsEps)
    {
        var preMoeNorm = _loader.LoadTensor($"{prefix}.pre_feedforward_layernorm.weight");
        var preMlp2Norm = _loader.LoadTensor($"{prefix}.pre_feedforward_layernorm_2.weight");
        var postMoeNorm = _loader.LoadTensor($"{prefix}.post_feedforward_layernorm_1.weight");
        var postMlp2Norm = _loader.LoadTensor($"{prefix}.post_feedforward_layernorm_2.weight");
        var postCombinedNorm = _loader.LoadTensor($"{prefix}.post_feedforward_layernorm.weight");

        // MoE branch
        var moeIn = TensorOperations.RmsNorm(attnResidual, preMoeNorm, rmsEps);
        var routerProj = _loader.LoadTensor($"{prefix}.router.proj.weight");
        var routerScale = _loader.TryLoadTensor($"{prefix}.router.scale");
        var perExpertScale = _loader.TryLoadTensor($"{prefix}.router.per_expert_scale");
        var expertsGateUp = _loader.LoadTensor($"{prefix}.experts.gate_up_proj");
        var expertsDown = _loader.LoadTensor($"{prefix}.experts.down_proj");

        var topK = _config.TextConfig?.TopKExperts ?? 1;
        var moeOut = MoeBlock.Forward(
            moeIn, routerProj, routerScale, perExpertScale,
            expertsGateUp, expertsDown, topK);
        moeOut = TensorOperations.RmsNorm(moeOut, postMoeNorm, rmsEps);

        // Dense shared (mlp2) branch
        var mlp2In = TensorOperations.RmsNorm(attnResidual, preMlp2Norm, rmsEps);
        var gateWeight = _loader.LoadTensor($"{prefix}.mlp.gate_proj.weight");
        var upWeight = _loader.LoadTensor($"{prefix}.mlp.up_proj.weight");
        var downWeight = _loader.LoadTensor($"{prefix}.mlp.down_proj.weight");
        var mlp2Out = FeedForward.Forward(mlp2In, gateWeight, upWeight, downWeight);
        mlp2Out = TensorOperations.RmsNorm(mlp2Out, postMlp2Norm, rmsEps);

        // Combine + outer norm
        var combined = moeOut + mlp2Out;
        return TensorOperations.RmsNorm(combined, postCombinedNorm, rmsEps);
    }

    /// <summary>
    /// Resets the KV cache, typically called at the start of a new generation session.
    /// </summary>
    public void ResetCache()
    {
        _kvCache.Clear();
    }
}
