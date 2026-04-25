using System;
using WebExpress.LLM.Tensor;

namespace WebExpress.LLM.Gemma;

/// <summary>
/// Implements the Mixture-of-Experts (MoE) sublayer used in Gemma-4 MoE variants.
/// </summary>
/// <remarks>
/// Router pipeline (mirrors <c>gemma/gm/nn/gemma4/_moe.MoERagged.__call__</c>):
/// <code>
///   router_input = rms_norm(x, with_scale=False)        // scale-less RMSNorm
///   router_input = router_input * rsqrt(features) * router_scale  // router_scale: [features]
///   logits       = router_input @ router_proj^T          // [seqLen, numExperts]
///   probs        = softmax(logits, axis=-1)              // full distribution
///   choices      = topk(probs, k)
///   weights[k]   = probs[choice_k] / sum_{i in topK} probs[choice_i]
/// </code>
/// For each chosen expert, the token is passed through a SwiGLU-style feed-forward
/// network. The output is scaled by the routing weight and (optionally) by a
/// learned <c>per_expert_scale</c>; per-expert scaling is applied to the routing
/// weight, which is mathematically equivalent to scaling the expert output before
/// the weighted sum.
///
/// Weight layout (HuggingFace convention, [out, in] per-expert, stacked along axis 0):
///   router.proj.weight            : [numExperts, hiddenSize]
///   router.scale                  : [hiddenSize], optional (per-feature router scale)
///   router.per_expert_scale       : [numExperts], optional
///   experts.gate_up_proj          : [numExperts, 2 * moeIntermediateSize, hiddenSize]
///   experts.down_proj             : [numExperts, hiddenSize, moeIntermediateSize]
/// </remarks>
public static class MoeBlock
{
    /// <summary>
    /// Runs the MoE forward pass for a batch of tokens.
    /// </summary>
    /// <param name="hidden">Input activations of shape [seqLen, hiddenSize].</param>
    /// <param name="routerProjWeight">Router projection of shape [numExperts, hiddenSize].</param>
    /// <param name="routerScale">Optional per-feature router scale of shape [hiddenSize], applied after a scale-less RMSNorm and the <c>1/sqrt(features)</c> factor.</param>
    /// <param name="perExpertScale">Optional per-expert weight, shape [numExperts]. Applied to the routing weight (equivalent to scaling the expert output).</param>
    /// <param name="expertsGateUpProj">Fused gate+up expert weights of shape [numExperts, 2*moeInter, hiddenSize].</param>
    /// <param name="expertsDownProj">Expert down-projection weights of shape [numExperts, hiddenSize, moeInter].</param>
    /// <param name="topK">The number of experts to activate per token.</param>
    /// <param name="rmsNormEpsilon">Epsilon used by the scale-less router RMSNorm.</param>
    /// <returns>Output tensor of shape [seqLen, hiddenSize].</returns>
    public static Tensor.Tensor Forward(
        Tensor.Tensor hidden,
        Tensor.Tensor routerProjWeight,
        Tensor.Tensor routerScale,
        Tensor.Tensor perExpertScale,
        Tensor.Tensor expertsGateUpProj,
        Tensor.Tensor expertsDownProj,
        int topK,
        float rmsNormEpsilon = 1e-6f)
    {
        ArgumentNullException.ThrowIfNull(hidden);
        ArgumentNullException.ThrowIfNull(routerProjWeight);
        ArgumentNullException.ThrowIfNull(expertsGateUpProj);
        ArgumentNullException.ThrowIfNull(expertsDownProj);

        if (hidden.Rank != 2)
        {
            throw new ArgumentException("Hidden input must be 2D [seqLen, hiddenSize].", nameof(hidden));
        }

        if (routerProjWeight.Rank != 2)
        {
            throw new ArgumentException("Router projection weight must be 2D [numExperts, hiddenSize].", nameof(routerProjWeight));
        }

        if (expertsGateUpProj.Rank != 3 || expertsDownProj.Rank != 3)
        {
            throw new ArgumentException("Expert weights must be 3D [numExperts, out, in].");
        }

        var seqLen = hidden.Shape[0];
        var hiddenSize = hidden.Shape[1];
        var numExperts = routerProjWeight.Shape[0];
        var gateUpOut = expertsGateUpProj.Shape[1];
        var moeInter = gateUpOut / 2;

        if (gateUpOut % 2 != 0)
        {
            throw new ArgumentException(
                $"Fused gate_up_proj output dimension {gateUpOut} must be even.");
        }

        if (expertsGateUpProj.Shape[0] != numExperts || expertsDownProj.Shape[0] != numExperts)
        {
            throw new ArgumentException("Expert weight first dimension must match router numExperts.");
        }

        if (topK <= 0 || topK > numExperts)
        {
            throw new ArgumentOutOfRangeException(nameof(topK),
                $"topK must be in range [1, {numExperts}].");
        }

        if (routerScale != null && routerScale.Length != hiddenSize)
        {
            throw new ArgumentException(
                $"router.scale length {routerScale.Length} must match hidden size {hiddenSize}.",
                nameof(routerScale));
        }

        // Router input: scale-less RMSNorm, then (1/sqrt(features)) * router_scale.
        var routerInput = TensorOperations.RmsNorm(hidden, weight: null, rmsNormEpsilon);
        var rsqrtFeatures = 1.0f / MathF.Sqrt(hiddenSize);

        if (routerScale != null)
        {
            for (var t = 0; t < seqLen; t++)
            {
                for (var f = 0; f < hiddenSize; f++)
                {
                    routerInput[t, f] = routerInput[t, f] * rsqrtFeatures * routerScale[f];
                }
            }
        }
        else
        {
            // No router_scale weight: just apply the rsqrt(features) factor.
            for (var i = 0; i < routerInput.Length; i++)
            {
                routerInput.Data[i] *= rsqrtFeatures;
            }
        }

        // Router logits: [seqLen, numExperts] = routerInput @ router.proj.weight^T
        var routerLogits = TensorOperations.MatMul(routerInput, routerProjWeight.Transpose());

        var output = new Tensor.Tensor(seqLen, hiddenSize);

        // Scratch buffers reused across tokens
        var tokenLogits = new float[numExperts];
        var tokenProbs = new float[numExperts];
        var topIdx = new int[topK];
        var topVals = new float[topK];

        for (var t = 0; t < seqLen; t++)
        {
            for (var e = 0; e < numExperts; e++)
            {
                tokenLogits[e] = routerLogits[t, e];
            }

            // Full softmax over all experts (router_probs in the reference).
            FullSoftmax(tokenLogits, tokenProbs);

            // Top-k selection over the probabilities (rank-equivalent to top-k of logits).
            SelectTopK(tokenProbs, topIdx, topVals);

            // Renormalize: weights[k] = probs[choice_k] / sum_{i in topK} probs[choice_i].
            // If sum is zero (only possible for masked/padded tokens) keep zeros.
            var sum = 0f;

            for (var i = 0; i < topK; i++)
            {
                sum += topVals[i];
            }

            if (sum > 0f)
            {
                for (var i = 0; i < topK; i++)
                {
                    topVals[i] /= sum;
                }
            }

            if (perExpertScale != null && perExpertScale.Length == numExperts)
            {
                for (var i = 0; i < topK; i++)
                {
                    topVals[i] *= perExpertScale[topIdx[i]];
                }
            }

            for (var i = 0; i < topK; i++)
            {
                var expertIdx = topIdx[i];
                var weight = topVals[i];

                if (weight == 0f)
                {
                    continue;
                }

                AccumulateExpert(
                    hidden, t, expertIdx, weight,
                    expertsGateUpProj, expertsDownProj,
                    hiddenSize, moeInter,
                    output);
            }
        }

        return output;
    }

    /// <summary>
    /// Numerically stable softmax, written into <paramref name="probs"/>.
    /// </summary>
    private static void FullSoftmax(float[] logits, float[] probs)
    {
        var max = float.NegativeInfinity;

        for (var i = 0; i < logits.Length; i++)
        {
            if (logits[i] > max)
            {
                max = logits[i];
            }
        }

        var sum = 0f;

        for (var i = 0; i < logits.Length; i++)
        {
            probs[i] = MathF.Exp(logits[i] - max);
            sum += probs[i];
        }

        if (sum > 0f)
        {
            for (var i = 0; i < probs.Length; i++)
            {
                probs[i] /= sum;
            }
        }
    }

    /// <summary>
    /// Computes one expert's SwiGLU feed-forward for a single token and adds
    /// the weighted result to the output row.
    /// </summary>
    private static void AccumulateExpert(
        Tensor.Tensor hidden, int tokenIdx, int expertIdx, float weight,
        Tensor.Tensor expertsGateUpProj, Tensor.Tensor expertsDownProj,
        int hiddenSize, int moeInter,
        Tensor.Tensor output)
    {
        var hiddenData = hidden.Data;
        var gateUpData = expertsGateUpProj.Data;
        var downData = expertsDownProj.Data;
        var outData = output.Data;

        var hiddenOffset = tokenIdx * hiddenSize;
        var gateUpOut = 2 * moeInter;

        // Per-expert slice offsets
        var gateUpExpertStride = gateUpOut * hiddenSize;
        var downExpertStride = hiddenSize * moeInter;

        var gateUpBase = expertIdx * gateUpExpertStride;
        var downBase = expertIdx * downExpertStride;

        // 1. Compute gate_up = hidden_token @ gate_up_proj[expert]^T  -> length 2*moeInter
        //    gate_up_proj[expert] has shape [2*moeInter, hiddenSize] (HF [out, in] layout).
        var gateUpRow = new float[gateUpOut];

        for (var o = 0; o < gateUpOut; o++)
        {
            var rowOffset = gateUpBase + o * hiddenSize;
            var sum = 0f;

            for (var h = 0; h < hiddenSize; h++)
            {
                sum += hiddenData[hiddenOffset + h] * gateUpData[rowOffset + h];
            }

            gateUpRow[o] = sum;
        }

        // 2. Split fused output into gate and up halves, apply GELU(gate) * up.
        var activated = new float[moeInter];
        var sqrt2OverPi = MathF.Sqrt(2.0f / MathF.PI);

        for (var m = 0; m < moeInter; m++)
        {
            var gate = gateUpRow[m];
            var up = gateUpRow[moeInter + m];

            // gelu_pytorch_tanh approximation, matching TensorOperations.Gelu.
            var inner = sqrt2OverPi * (gate + 0.044715f * gate * gate * gate);
            var gelu = 0.5f * gate * (1.0f + MathF.Tanh(inner));

            activated[m] = gelu * up;
        }

        // 3. Down-project: out = activated @ down_proj[expert]^T  -> length hiddenSize
        //    down_proj[expert] has shape [hiddenSize, moeInter].
        //    Accumulate weighted result directly into output[t, :].
        var outOffset = tokenIdx * hiddenSize;

        for (var h = 0; h < hiddenSize; h++)
        {
            var rowOffset = downBase + h * moeInter;
            var sum = 0f;

            for (var m = 0; m < moeInter; m++)
            {
                sum += activated[m] * downData[rowOffset + m];
            }

            outData[outOffset + h] += weight * sum;
        }
    }

    /// <summary>
    /// Fills <paramref name="idx"/> and <paramref name="vals"/> with the indices and
    /// values of the top-k largest entries of <paramref name="logits"/> (unsorted order).
    /// </summary>
    private static void SelectTopK(float[] logits, int[] idx, float[] vals)
    {
        var k = idx.Length;

        // Initialise with the first k entries
        for (var i = 0; i < k; i++)
        {
            idx[i] = i;
            vals[i] = logits[i];
        }

        // Find current minimum of the top-k window
        var minPos = 0;

        for (var i = 1; i < k; i++)
        {
            if (vals[i] < vals[minPos])
            {
                minPos = i;
            }
        }

        // Replace-min for the remaining entries
        for (var i = k; i < logits.Length; i++)
        {
            var v = logits[i];

            if (v > vals[minPos])
            {
                vals[minPos] = v;
                idx[minPos] = i;

                // Rescan for new minimum
                minPos = 0;

                for (var j = 1; j < k; j++)
                {
                    if (vals[j] < vals[minPos])
                    {
                        minPos = j;
                    }
                }
            }
        }
    }

}
