using System;
using System.Threading.Tasks;
using WebExpress.LLM.Tensor;

namespace WebExpress.LLM.Gemma;

/// <summary>
/// Implements the multi-head attention mechanism used in Gemma-4 transformer layers,
/// supporting both full attention and sliding window attention patterns.
/// </summary>
/// <remarks>
/// Gemma-4 uses grouped-query attention (GQA) where a smaller number of key-value heads
/// are shared across multiple query heads. It alternates between sliding window attention
/// (512-token window) and full causal attention across its 35 layers.
/// </remarks>
public sealed class MultiHeadAttention
{
    /// <summary>
    /// Large negative value used to mask attention logits. Matches the Gemma
    /// reference (<c>gemma/gm/nn/gemma4/_modules.py</c>, <c>K_MASK</c>). Using a
    /// finite value rather than <see cref="float.NegativeInfinity"/> keeps the
    /// softmax well-defined even when an entire row is masked (e.g. an empty
    /// sliding window with bfloat16 later on).
    /// </summary>
    public const float MaskValue = -2.3819763e38f;

    private readonly int _numQueryHeads;
    private readonly int _numKvHeads;
    private readonly int _headDim;
    private readonly bool _isFullAttention;
    private readonly int _slidingWindowSize;
    private readonly RotaryEmbedding _rope;
    private readonly float _attentionLogitsSoftcap;

    /// <summary>
    /// Initializes a new MultiHeadAttention layer.
    /// </summary>
    /// <param name="numQueryHeads">Number of query attention heads.</param>
    /// <param name="numKvHeads">Number of key-value attention heads (for GQA).</param>
    /// <param name="headDim">Dimension of each attention head.</param>
    /// <param name="isFullAttention">Whether this layer uses full attention (true) or sliding window (false).</param>
    /// <param name="slidingWindowSize">The sliding window size for local attention.</param>
    /// <param name="rope">The rotary position embedding to apply to queries and keys.</param>
    /// <param name="attentionLogitsSoftcap">
    /// Optional soft-cap applied to pre-softmax attention scores via
    /// <c>tanh(score / cap) * cap</c>. A value of 0 (the default) disables the
    /// soft cap. Used by Gemma-2 and some Gemma-4 variants to stabilise training.
    /// </param>
    public MultiHeadAttention(
        int numQueryHeads,
        int numKvHeads,
        int headDim,
        bool isFullAttention,
        int slidingWindowSize,
        RotaryEmbedding rope,
        float attentionLogitsSoftcap = 0f)
    {
        _numQueryHeads = numQueryHeads;
        _numKvHeads = numKvHeads;
        _headDim = headDim;
        _isFullAttention = isFullAttention;
        _slidingWindowSize = slidingWindowSize;
        _rope = rope ?? throw new ArgumentNullException(nameof(rope));
        _attentionLogitsSoftcap = attentionLogitsSoftcap;
    }

    /// <summary>
    /// Computes the multi-head attention forward pass.
    /// </summary>
    /// <param name="input">Input tensor of shape [seqLen, hiddenSize].</param>
    /// <param name="qProjWeight">Query projection weight [numQueryHeads * headDim, hiddenSize].</param>
    /// <param name="kProjWeight">Key projection weight [numKvHeads * headDim, hiddenSize].</param>
    /// <param name="vProjWeight">Value projection weight [numKvHeads * headDim, hiddenSize].</param>
    /// <param name="oProjWeight">Output projection weight [hiddenSize, numQueryHeads * headDim].</param>
    /// <param name="kvCache">Optional KV cache for autoregressive generation.</param>
    /// <param name="layerIndex">The layer index (used for KV cache keying).</param>
    /// <param name="qNormWeight">Optional per-head query RMSNorm weight of shape [qHeadDim], applied before RoPE.</param>
    /// <param name="kNormWeight">Optional per-head key RMSNorm weight of shape [kHeadDim], applied before RoPE.</param>
    /// <param name="rmsNormEpsilon">Epsilon used for the optional q/k RMSNorm operations.</param>
    /// <returns>The attention output of shape [seqLen, hiddenSize].</returns>
    /// <remarks>
    /// The Gemma-4 reference (<c>gemma/gm/nn/gemma4/_modules.Attention</c>) applies a
    /// scale-less <c>value_norm</c> to V after the projection, before the KV-cache
    /// update. It also computes the attention logits as a plain
    /// <c>einsum('BTNH,BSNH-&gt;BTNS', q, k)</c> with no <c>1/sqrt(head_dim)</c> factor —
    /// magnitude is regulated by the learnable <c>q_norm</c>/<c>k_norm</c> scales.
    /// </remarks>
    public Tensor.Tensor Forward(
        Tensor.Tensor input,
        Tensor.Tensor qProjWeight,
        Tensor.Tensor kProjWeight,
        Tensor.Tensor vProjWeight,
        Tensor.Tensor oProjWeight,
        KvCache kvCache = null,
        int layerIndex = 0,
        Tensor.Tensor qNormWeight = null,
        Tensor.Tensor kNormWeight = null,
        float rmsNormEpsilon = 1e-6f)
    {
        var seqLen = input.Shape[0];
        var hiddenSize = input.Shape[1];

        //System.Console.WriteLine($"MultiHeadAttention.Forward: seqLen={seqLen}, hiddenSize={hiddenSize}, " +
        //    $"numQueryHeads={_numQueryHeads}, numKvHeads={_numKvHeads}, headDim={_headDim}, " +
        //    $"isFullAttention={_isFullAttention}, slidingWindowSize={_slidingWindowSize}");

        // Project to Q, K, V: [seqLen, hiddenSize] × [hiddenSize, numHeads*headDim]
        var qProj = TensorOperations.MatMul(input, Transpose2D(qProjWeight));
        var kProj = TensorOperations.MatMul(input, Transpose2D(kProjWeight));
        var vProj = TensorOperations.MatMul(input, Transpose2D(vProjWeight));

        // Derive actual head dimensions from the projected tensor shapes
        // to avoid mismatches between config values and real weight sizes.
        // In Gemma-4 full attention layers with attention_k_eq_v:
        //   Q uses global_head_dim (e.g. 512) per query head
        //   K uses head_dim (e.g. 128) per KV head
        //   V shares K's weight, so also uses head_dim per KV head
        //   O expects numQueryHeads * global_head_dim (matches Q)
        // The gap between vHeadDim and the o_proj expected dimension is
        // bridged by concatenating unused Q dimensions ("pass-through").
        var qProjDim = qProj.Shape[1];
        var kProjDim = kProj.Shape[1];
        var vProjDim = vProj.Shape[1];

        if (qProjDim % _numQueryHeads != 0)
        {
            throw new InvalidOperationException(
                $"Q projection dimension {qProjDim} is not evenly divisible by the number of query heads {_numQueryHeads}.");
        }

        if (kProjDim % _numKvHeads != 0)
        {
            throw new InvalidOperationException(
                $"K projection dimension {kProjDim} is not evenly divisible by the number of KV heads {_numKvHeads}.");
        }

        if (vProjDim % _numKvHeads != 0)
        {
            throw new InvalidOperationException(
                $"V projection dimension {vProjDim} is not evenly divisible by the number of KV heads {_numKvHeads}.");
        }

        var qHeadDim = qProjDim / _numQueryHeads;
        var kHeadDim = kProjDim / _numKvHeads;
        var vHeadDim = vProjDim / _numKvHeads;

        // Derive the expected per-head output dimension from the o_proj weight.
        // o_proj weight shape: [hiddenSize, numQueryHeads * outputHeadDim]
        var oProjInputDim = oProjWeight.Shape[1];

        if (oProjInputDim % _numQueryHeads != 0)
        {
            throw new InvalidOperationException(
                $"O projection input dimension {oProjInputDim} is not evenly divisible by the number of query heads {_numQueryHeads}.");
        }

        var outputHeadDim = oProjInputDim / _numQueryHeads;

        // Reshape to [numHeads, seqLen, headDim]
        var Q = ReshapeToHeads(qProj, _numQueryHeads, seqLen, qHeadDim);
        var K = ReshapeToHeads(kProj, _numKvHeads, seqLen, kHeadDim);
        var V = ReshapeToHeads(vProj, _numKvHeads, seqLen, vHeadDim);

        // Per-head RMSNorm on Q and K (Gemma-4 uses q_norm/k_norm before RoPE).
        // RmsNorm normalises over the last dimension, which is the head dimension here.
        if (qNormWeight != null)
        {
            Q = TensorOperations.RmsNorm(Q, qNormWeight, rmsNormEpsilon);
        }

        if (kNormWeight != null)
        {
            K = TensorOperations.RmsNorm(K, kNormWeight, rmsNormEpsilon);
        }

        // Scale-less value_norm. Always applied in the Gemma-4 reference
        // (`_layers.RMSNorm(with_scale=False)`); there is no learnable
        // weight in the checkpoint for this norm.
        V = TensorOperations.RmsNorm(V, weight: null, rmsNormEpsilon);

        // Apply RoPE to Q and K
        var startPosition = kvCache?.GetSequenceLength(layerIndex) ?? 0;
        Q = _rope.Apply(Q, startPosition);
        K = _rope.Apply(K, startPosition);

        // Update KV cache
        if (kvCache != null)
        {
            kvCache.Update(layerIndex, K, V);
            var cached = kvCache.Get(layerIndex);
            K = cached.Keys;
            V = cached.Values;
        }

        // Handle GQA: repeat K,V heads to match Q heads
        if (_numKvHeads < _numQueryHeads)
        {
            var repeatFactor = _numQueryHeads / _numKvHeads;
            K = RepeatKvHeads(K, repeatFactor);
            V = RepeatKvHeads(V, repeatFactor);
        }

        // Compute attention scores: Q @ K^T / sqrt(dotDim)
        // When Q has a larger head dimension than K (asymmetric head dims),
        // the dot product uses only the first kHeadDim dimensions of Q.
        var kvSeqLen = K.Shape[1];
        var scores = ComputeAttentionScores(
            Q, K, _numQueryHeads, seqLen, kvSeqLen, qHeadDim, kHeadDim,
            _attentionLogitsSoftcap);

        // Apply attention mask
        ApplyMask(scores, seqLen, kvSeqLen, startPosition);

        // Softmax over last dimension
        scores = TensorOperations.Softmax(scores);

        // Attention output: scores @ V (result has vHeadDim per head)
        var attnOutput = ComputeAttentionOutput(scores, V, _numQueryHeads, seqLen, kvSeqLen, vHeadDim);

        // When V's per-head dimension (vHeadDim) is smaller than what o_proj
        // expects (outputHeadDim), Q has extra dimensions beyond those used
        // in the dot product with K.  These "pass-through" dimensions are
        // concatenated with the attention output so the combined per-head
        // dimension matches o_proj.  This occurs in Gemma-4 full attention
        // layers when attention_k_eq_v is true (V = K, both use headDim)
        // while Q and o_proj use the larger globalHeadDim.
        if (vHeadDim < outputHeadDim)
        {
            var dotDim = Math.Min(qHeadDim, kHeadDim);
            var passThroughDim = outputHeadDim - vHeadDim;
            attnOutput = ConcatQPassThrough(attnOutput, Q, _numQueryHeads, seqLen, vHeadDim, dotDim, passThroughDim);
        }

        // Reshape from [numQueryHeads, seqLen, outputHeadDim] -> [seqLen, numQueryHeads * outputHeadDim]
        var concatenated = ReshapeFromHeads(attnOutput, _numQueryHeads, seqLen, outputHeadDim);

        // Output projection
        var output = TensorOperations.MatMul(concatenated, Transpose2D(oProjWeight));

        return output;
    }

    /// <summary>
    /// Returns a new tensor with the rows and columns of the specified 2D tensor exchanged.
    /// </summary>
    /// <param name="t">The 2D tensor to transpose. Must have a rank of 2.</param>
    /// <returns>A tensor representing the transposed version of the input tensor.</returns>
    /// <exception cref="ArgumentException">Thrown if the input tensor does not have a rank of 2.</exception>
    private static Tensor.Tensor Transpose2D(Tensor.Tensor t)
    {
        if (t.Rank != 2) throw new ArgumentException("Transpose2D requires 2D tensor.");
        return t.Transpose();
    }

    /// <summary>
    /// Reshapes [seqLen, numHeads * headDim] to [numHeads, seqLen, headDim].
    /// </summary>
    private static Tensor.Tensor ReshapeToHeads(Tensor.Tensor input, int numHeads, int seqLen, int headDim)
    {
        var data = input.Data;
        var expectedLength = seqLen * numHeads * headDim;

        if (data.Length < expectedLength)
        {
            throw new ArgumentException(
                $"Input tensor length {data.Length} is too small for the requested reshape " +
                $"(seqLen={seqLen}, numHeads={numHeads}, headDim={headDim}, expected={expectedLength}).");
        }

        var result = new float[numHeads * seqLen * headDim];

        for (var s = 0; s < seqLen; s++)
        {
            for (var h = 0; h < numHeads; h++)
            {
                var srcOffset = s * numHeads * headDim + h * headDim;
                var dstOffset = h * seqLen * headDim + s * headDim;
                Array.Copy(data, srcOffset, result, dstOffset, headDim);
            }
        }

        return new Tensor.Tensor([numHeads, seqLen, headDim], result);
    }

    /// <summary>
    /// Reshapes [numHeads, seqLen, headDim] to [seqLen, numHeads * headDim].
    /// </summary>
    private static Tensor.Tensor ReshapeFromHeads(Tensor.Tensor input, int numHeads, int seqLen, int headDim)
    {
        var data = input.Data;
        var result = new float[seqLen * numHeads * headDim];

        for (var s = 0; s < seqLen; s++)
        {
            for (var h = 0; h < numHeads; h++)
            {
                var srcOffset = h * seqLen * headDim + s * headDim;
                var dstOffset = s * numHeads * headDim + h * headDim;
                Array.Copy(data, srcOffset, result, dstOffset, headDim);
            }
        }

        return new Tensor.Tensor([seqLen, numHeads * headDim], result);
    }

    /// <summary>
    /// Concatenates Q pass-through dimensions with the attention output.
    /// When V's per-head dimension is smaller than o_proj's expected input,
    /// Q has unused dimensions (beyond those used for the K dot product)
    /// that are appended to the attention output for each head.
    /// </summary>
    private static Tensor.Tensor ConcatQPassThrough(
        Tensor.Tensor attnOutput, Tensor.Tensor Q,
        int numHeads, int seqLen, int vHeadDim, int dotDim, int passThroughDim)
    {
        var outputHeadDim = vHeadDim + passThroughDim;
        var qHeadDim = Q.Shape[2];
        var result = new float[numHeads * seqLen * outputHeadDim];

        for (var h = 0; h < numHeads; h++)
        {
            for (var s = 0; s < seqLen; s++)
            {
                var attnSrc = h * seqLen * vHeadDim + s * vHeadDim;
                var qSrc = h * seqLen * qHeadDim + s * qHeadDim + dotDim;
                var dst = h * seqLen * outputHeadDim + s * outputHeadDim;

                // Copy attention output (vHeadDim dimensions)
                Array.Copy(attnOutput.Data, attnSrc, result, dst, vHeadDim);

                // Copy Q pass-through (passThroughDim dimensions starting at dotDim)
                Array.Copy(Q.Data, qSrc, result, dst + vHeadDim, passThroughDim);
            }
        }

        return new Tensor.Tensor([numHeads, seqLen, outputHeadDim], result);
    }

    /// <summary>
    /// Repeats KV heads to match the number of query heads (for grouped-query attention).
    /// </summary>
    private static Tensor.Tensor RepeatKvHeads(Tensor.Tensor kv, int repeatFactor)
    {
        if (repeatFactor == 1) return kv;

        var numKvHeads = kv.Shape[0];
        var seqLen = kv.Shape[1];
        var headDim = kv.Shape[2];
        var newNumHeads = numKvHeads * repeatFactor;
        var result = new float[newNumHeads * seqLen * headDim];

        for (var h = 0; h < numKvHeads; h++)
        {
            for (var r = 0; r < repeatFactor; r++)
            {
                var srcOffset = h * seqLen * headDim;
                var dstOffset = (h * repeatFactor + r) * seqLen * headDim;
                Array.Copy(kv.Data, srcOffset, result, dstOffset, seqLen * headDim);
            }
        }

        return new Tensor.Tensor([newNumHeads, seqLen, headDim], result);
    }

    /// <summary>
    /// Computes Q @ K^T per head, optionally applying an attention-logits soft
    /// cap (<c>tanh(score / cap) * cap</c>) before the softmax. When Q and K
    /// have different per-head dimensions the dot product is computed over the
    /// smaller dimension (kHeadDim).
    /// </summary>
    /// <remarks>
    /// No <c>1/sqrt(head_dim)</c> factor is applied: the Gemma-4 reference
    /// uses a plain einsum and relies on the learnable <c>q_norm</c>/<c>k_norm</c>
    /// scales to control the score magnitude.
    /// </remarks>
    private static Tensor.Tensor ComputeAttentionScores(
        Tensor.Tensor Q, Tensor.Tensor K, int numHeads, int queryLen, int kvLen, int qHeadDim, int kHeadDim,
        float softcap)
    {
        var dotDim = Math.Min(qHeadDim, kHeadDim);
        var scores = new float[numHeads * queryLen * kvLen];
        var applySoftcap = softcap > 0f;

        Parallel.For(0, numHeads, h =>
        {
            var qOffset = h * queryLen * qHeadDim;
            var kOffset = h * kvLen * kHeadDim;
            var sOffset = h * queryLen * kvLen;

            for (var i = 0; i < queryLen; i++)
            {
                for (var j = 0; j < kvLen; j++)
                {
                    var dot = 0.0f;

                    for (var d = 0; d < dotDim; d++)
                    {
                        dot += Q.Data[qOffset + i * qHeadDim + d] * K.Data[kOffset + j * kHeadDim + d];
                    }

                    var score = dot;

                    if (applySoftcap)
                    {
                        score = MathF.Tanh(score / softcap) * softcap;
                    }

                    scores[sOffset + i * kvLen + j] = score;
                }
            }
        });

        return new Tensor.Tensor([numHeads, queryLen, kvLen], scores);
    }

    /// <summary>
    /// Applies causal and/or sliding window mask in place.
    /// </summary>
    private void ApplyMask(Tensor.Tensor scores, int queryLen, int kvLen, int startPosition)
    {
        var numHeads = scores.Shape[0];

        for (var h = 0; h < numHeads; h++)
        {
            for (var i = 0; i < queryLen; i++)
            {
                var queryPos = startPosition + i;

                for (var j = 0; j < kvLen; j++)
                {
                    // Causal mask: cannot attend to future positions
                    if (j > queryPos)
                    {
                        scores[h, i, j] = MaskValue;
                        continue;
                    }

                    // Sliding window mask (only for sliding attention layers)
                    if (!_isFullAttention && queryPos - j >= _slidingWindowSize)
                    {
                        scores[h, i, j] = MaskValue;
                    }
                }
            }
        }
    }

    /// <summary>
    /// Computes softmax(scores) @ V for each head.
    /// </summary>
    private static Tensor.Tensor ComputeAttentionOutput(
        Tensor.Tensor scores, Tensor.Tensor V, int numHeads, int queryLen, int kvLen, int headDim)
    {
        var result = new float[numHeads * queryLen * headDim];

        for (var h = 0; h < numHeads; h++)
        {
            var sOffset = h * queryLen * kvLen;
            var vOffset = h * kvLen * headDim;
            var rOffset = h * queryLen * headDim;

            for (var i = 0; i < queryLen; i++)
            {
                for (var d = 0; d < headDim; d++)
                {
                    var sum = 0.0f;

                    for (var j = 0; j < kvLen; j++)
                    {
                        sum += scores.Data[sOffset + i * kvLen + j] * V.Data[vOffset + j * headDim + d];
                    }

                    result[rOffset + i * headDim + d] = sum;
                }
            }
        }

        return new Tensor.Tensor([numHeads, queryLen, headDim], result);
    }
}
