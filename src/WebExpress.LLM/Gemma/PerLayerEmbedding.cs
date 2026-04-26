using System;
using WebExpress.LLM.Tensor;

namespace WebExpress.LLM.Gemma;

/// <summary>
/// Implements the Per-Layer Embedding (PLE) pipeline used by Gemma-4 variants
/// where <c>hidden_size_per_layer_input &gt; 0</c>.
/// </summary>
/// <remarks>
/// Mirrors the vLLM reference (<c>vllm/model_executor/models/gemma4.py</c>):
/// <list type="bullet">
///   <item><description><see cref="BuildPerLayerInputs"/> mirrors
///   <c>Gemma4SelfDecoderLayers.get_per_layer_inputs</c> +
///   <c>project_per_layer_inputs</c> (lines 832-885).</description></item>
///   <item><description><see cref="ApplyPerLayerContribution"/> mirrors the
///   per-layer block in <c>Gemma4DecoderLayer.forward</c> (lines 740-748).</description></item>
/// </list>
/// </remarks>
public static class PerLayerEmbedding
{
    /// <summary>
    /// Computes the combined per-layer inputs that feed each decoder layer's
    /// PLE block.
    /// </summary>
    /// <remarks>
    /// Steps:
    /// <list type="number">
    ///   <item><description>Embed each input token with the dedicated
    ///   <c>embed_tokens_per_layer</c> table (token IDs outside
    ///   <paramref name="vocabSizePerLayerInput"/> are zeroed).</description></item>
    ///   <item><description>Scale by <c>sqrt(hidden_size_per_layer_input)</c>.</description></item>
    ///   <item><description>Project the main hidden states with
    ///   <c>per_layer_model_projection</c> and scale by <c>hidden_size^-0.5</c>.</description></item>
    ///   <item><description>Apply <c>per_layer_projection_norm</c> on the
    ///   reshaped projection.</description></item>
    ///   <item><description>Combine: <c>(projection + per_layer_embeds) * (1/sqrt(2))</c>.</description></item>
    /// </list>
    /// </remarks>
    /// <param name="inputsEmbeds">Token embeddings of shape <c>[seqLen, hiddenSize]</c>
    /// (already scaled by <c>sqrt(hidden_size)</c> by the caller).</param>
    /// <param name="tokenIds">The original token IDs (length = seqLen).</param>
    /// <param name="embedTokensPerLayer">PLE embedding table of shape
    /// <c>[vocabSizePerLayerInput, numLayers * hiddenSizePerLayerInput]</c>.</param>
    /// <param name="perLayerModelProjection">Projection of shape
    /// <c>[numLayers * hiddenSizePerLayerInput, hiddenSize]</c>.</param>
    /// <param name="perLayerProjectionNormWeight">RMSNorm weight of shape
    /// <c>[hiddenSizePerLayerInput]</c>.</param>
    /// <param name="hiddenSize">The model hidden size.</param>
    /// <param name="numLayers">The number of decoder layers.</param>
    /// <param name="hiddenSizePerLayerInput">The PLE per-layer dimension.</param>
    /// <param name="vocabSizePerLayerInput">The PLE vocabulary size; tokens
    /// outside this range produce a zero PLE embedding.</param>
    /// <param name="rmsEps">Epsilon used by the RMSNorm stage.</param>
    /// <returns>A tensor of shape
    /// <c>[seqLen, numLayers, hiddenSizePerLayerInput]</c>.</returns>
    public static Tensor.Tensor BuildPerLayerInputs(
        Tensor.Tensor inputsEmbeds,
        int[] tokenIds,
        Tensor.Tensor embedTokensPerLayer,
        Tensor.Tensor perLayerModelProjection,
        Tensor.Tensor perLayerProjectionNormWeight,
        int hiddenSize,
        int numLayers,
        int hiddenSizePerLayerInput,
        int vocabSizePerLayerInput,
        float rmsEps = 1e-6f)
    {
        ArgumentNullException.ThrowIfNull(inputsEmbeds);
        ArgumentNullException.ThrowIfNull(tokenIds);
        ArgumentNullException.ThrowIfNull(embedTokensPerLayer);
        ArgumentNullException.ThrowIfNull(perLayerModelProjection);
        ArgumentNullException.ThrowIfNull(perLayerProjectionNormWeight);

        if (numLayers <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(numLayers));
        }

        if (hiddenSizePerLayerInput <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(hiddenSizePerLayerInput));
        }

        var seqLen = tokenIds.Length;
        var totalPleDim = numLayers * hiddenSizePerLayerInput;

        // 1. Per-layer token embeddings (masked + scaled).
        var embedScale = MathF.Sqrt(hiddenSizePerLayerInput);
        var perLayerEmbeds = new float[seqLen * totalPleDim];
        var pleData = embedTokensPerLayer.Data;

        for (var i = 0; i < seqLen; i++)
        {
            var id = tokenIds[i];

            if (id < 0 || id >= vocabSizePerLayerInput)
            {
                // Out-of-range / masked token → zero PLE row.
                continue;
            }

            var srcOffset = id * totalPleDim;
            var dstOffset = i * totalPleDim;

            for (var d = 0; d < totalPleDim; d++)
            {
                perLayerEmbeds[dstOffset + d] = pleData[srcOffset + d] * embedScale;
            }
        }

        // 2. Project hidden states: [seqLen, hiddenSize] @ proj^T → [seqLen, totalPleDim]
        //    proj has HF [out, in] layout, so transpose for MatMul.
        var projection = TensorOperations.MatMul(inputsEmbeds, perLayerModelProjection.Transpose());

        // 3. Scale projection by hidden_size^-0.5
        var projectionScale = 1.0f / MathF.Sqrt(hiddenSize);
        projection *= projectionScale;

        // 4. Reshape to [seqLen, numLayers, hiddenSizePerLayerInput] and RMSNorm
        //    along the last dim.
        var reshaped = projection.Reshape(seqLen, numLayers, hiddenSizePerLayerInput);
        var normalised = TensorOperations.RmsNorm(reshaped, perLayerProjectionNormWeight, rmsEps);

        // 5. Combine: (projection + per_layer_embeds) * (1/sqrt(2))
        var combinedScale = 1.0f / MathF.Sqrt(2.0f);
        var normData = normalised.Data;
        var combined = new float[seqLen * totalPleDim];

        for (var i = 0; i < combined.Length; i++)
        {
            combined[i] = (normData[i] + perLayerEmbeds[i]) * combinedScale;
        }

        return new Tensor.Tensor([seqLen, numLayers, hiddenSizePerLayerInput], combined);
    }

    /// <summary>
    /// Applies the per-layer PLE contribution at the end of a decoder layer.
    /// Mirrors vLLM <c>Gemma4DecoderLayer.forward</c> lines 740-748.
    /// </summary>
    /// <remarks>
    /// <code>
    ///   gate         = per_layer_input_gate(hidden_states)
    ///   gate         = gelu_pytorch_tanh(gate)
    ///   gated        = gate * per_layer_input
    ///   contribution = per_layer_projection(gated)
    ///   contribution = post_per_layer_input_norm(contribution)
    ///   hidden       = hidden + contribution
    /// </code>
    /// </remarks>
    /// <param name="hidden">Hidden state of shape <c>[seqLen, hiddenSize]</c>.</param>
    /// <param name="perLayerInputForLayer">PLE input slice for this layer of
    /// shape <c>[seqLen, hiddenSizePerLayerInput]</c>.</param>
    /// <param name="inputGateWeight">Gate weight of shape
    /// <c>[hiddenSizePerLayerInput, hiddenSize]</c> (HF [out, in] layout).</param>
    /// <param name="projectionWeight">Projection weight of shape
    /// <c>[hiddenSize, hiddenSizePerLayerInput]</c> (HF [out, in] layout).</param>
    /// <param name="postNormWeight">RMSNorm weight of shape <c>[hiddenSize]</c>.</param>
    /// <param name="rmsEps">Epsilon for the RMSNorm stage.</param>
    /// <returns>Updated hidden state of shape <c>[seqLen, hiddenSize]</c>.</returns>
    public static Tensor.Tensor ApplyPerLayerContribution(
        Tensor.Tensor hidden,
        Tensor.Tensor perLayerInputForLayer,
        Tensor.Tensor inputGateWeight,
        Tensor.Tensor projectionWeight,
        Tensor.Tensor postNormWeight,
        float rmsEps = 1e-6f)
    {
        ArgumentNullException.ThrowIfNull(hidden);
        ArgumentNullException.ThrowIfNull(perLayerInputForLayer);
        ArgumentNullException.ThrowIfNull(inputGateWeight);
        ArgumentNullException.ThrowIfNull(projectionWeight);
        ArgumentNullException.ThrowIfNull(postNormWeight);

        // gate = hidden @ gate^T → [seqLen, hiddenSizePerLayerInput]
        var gate = TensorOperations.MatMul(hidden, inputGateWeight.Transpose());
        gate = TensorOperations.Gelu(gate);

        // gated element-wise
        var gated = gate * perLayerInputForLayer;

        // contribution = gated @ proj^T → [seqLen, hiddenSize]
        var contribution = TensorOperations.MatMul(gated, projectionWeight.Transpose());
        contribution = TensorOperations.RmsNorm(contribution, postNormWeight, rmsEps);

        return hidden + contribution;
    }

    /// <summary>
    /// Slices the per-layer inputs for a specific layer index.
    /// </summary>
    /// <param name="perLayerInputs">Tensor of shape
    /// <c>[seqLen, numLayers, hiddenSizePerLayerInput]</c>.</param>
    /// <param name="layerIndex">The decoder layer index (0-based).</param>
    /// <returns>A 2-D slice of shape <c>[seqLen, hiddenSizePerLayerInput]</c>.</returns>
    public static Tensor.Tensor SliceLayer(Tensor.Tensor perLayerInputs, int layerIndex)
    {
        ArgumentNullException.ThrowIfNull(perLayerInputs);

        if (perLayerInputs.Rank != 3)
        {
            throw new ArgumentException(
                $"Expected a 3-D tensor [seqLen, numLayers, perLayerDim], got rank {perLayerInputs.Rank}.",
                nameof(perLayerInputs));
        }

        var seqLen = perLayerInputs.Shape[0];
        var numLayers = perLayerInputs.Shape[1];
        var perLayerDim = perLayerInputs.Shape[2];

        if (layerIndex < 0 || layerIndex >= numLayers)
        {
            throw new ArgumentOutOfRangeException(nameof(layerIndex));
        }

        var slice = new float[seqLen * perLayerDim];
        var src = perLayerInputs.Data;

        for (var i = 0; i < seqLen; i++)
        {
            var srcOffset = (i * numLayers + layerIndex) * perLayerDim;
            var dstOffset = i * perLayerDim;
            Array.Copy(src, srcOffset, slice, dstOffset, perLayerDim);
        }

        return new Tensor.Tensor([seqLen, perLayerDim], slice);
    }
}
