using System;
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
    private readonly int _numQueryHeads;
    private readonly int _numKvHeads;
    private readonly int _headDim;
    private readonly bool _isFullAttention;
    private readonly int _slidingWindowSize;
    private readonly RotaryEmbedding _rope;

    /// <summary>
    /// Initializes a new MultiHeadAttention layer.
    /// </summary>
    /// <param name="numQueryHeads">Number of query attention heads.</param>
    /// <param name="numKvHeads">Number of key-value attention heads (for GQA).</param>
    /// <param name="headDim">Dimension of each attention head.</param>
    /// <param name="isFullAttention">Whether this layer uses full attention (true) or sliding window (false).</param>
    /// <param name="slidingWindowSize">The sliding window size for local attention.</param>
    /// <param name="rope">The rotary position embedding to apply to queries and keys.</param>
    public MultiHeadAttention(
        int numQueryHeads,
        int numKvHeads,
        int headDim,
        bool isFullAttention,
        int slidingWindowSize,
        RotaryEmbedding rope)
    {
        _numQueryHeads = numQueryHeads;
        _numKvHeads = numKvHeads;
        _headDim = headDim;
        _isFullAttention = isFullAttention;
        _slidingWindowSize = slidingWindowSize;
        _rope = rope ?? throw new ArgumentNullException(nameof(rope));
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
    /// <returns>The attention output of shape [seqLen, hiddenSize].</returns>
    public Tensor.Tensor Forward(
        Tensor.Tensor input,
        Tensor.Tensor qProjWeight,
        Tensor.Tensor kProjWeight,
        Tensor.Tensor vProjWeight,
        Tensor.Tensor oProjWeight,
        KvCache kvCache = null,
        int layerIndex = 0)
    {
        var seqLen = input.Shape[0];
        var hiddenSize = input.Shape[1];

        // Project to Q, K, V: [seqLen, hiddenSize] × [hiddenSize, numHeads*headDim]
        var qProj = TensorOperations.MatMul(input, Transpose2D(qProjWeight));
        var kProj = TensorOperations.MatMul(input, Transpose2D(kProjWeight));
        var vProj = TensorOperations.MatMul(input, Transpose2D(vProjWeight));

        // Derive actual head dimensions from the projected tensor shapes
        // to avoid mismatches between config values and real weight sizes.
        // In Gemma-4 full attention layers the query projection may use a
        // larger per-head dimension (global_head_dim) than the key/value
        // projections (head_dim), so each must be derived independently.
        var qProjDim = qProj.Shape[1];
        var kProjDim = kProj.Shape[1];

        if (qProjDim % _numQueryHeads != 0)
        {
            throw new InvalidOperationException(
                $"Query projection dimension {qProjDim} is not evenly divisible by the number of query heads {_numQueryHeads}.");
        }

        if (kProjDim % _numKvHeads != 0)
        {
            throw new InvalidOperationException(
                $"Key projection dimension {kProjDim} is not evenly divisible by the number of KV heads {_numKvHeads}.");
        }

        var qHeadDim = qProjDim / _numQueryHeads;
        var kvHeadDim = kProjDim / _numKvHeads;

        // Reshape to [numHeads, seqLen, headDim]
        var Q = ReshapeToHeads(qProj, _numQueryHeads, seqLen, qHeadDim);
        var K = ReshapeToHeads(kProj, _numKvHeads, seqLen, kvHeadDim);
        var V = ReshapeToHeads(vProj, _numKvHeads, seqLen, kvHeadDim);

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
        // the dot product uses only the first kvHeadDim dimensions of Q.
        var kvSeqLen = K.Shape[1];
        var scores = ComputeAttentionScores(Q, K, _numQueryHeads, seqLen, kvSeqLen, qHeadDim, kvHeadDim);

        // Apply attention mask
        ApplyMask(scores, seqLen, kvSeqLen, startPosition);

        // Softmax over last dimension
        scores = TensorOperations.Softmax(scores);

        // Attention output: scores @ V (result has kvHeadDim per head)
        var attnOutput = ComputeAttentionOutput(scores, V, _numQueryHeads, seqLen, kvSeqLen, kvHeadDim);

        // Reshape from [numQueryHeads, seqLen, kvHeadDim] -> [seqLen, numQueryHeads * kvHeadDim]
        var concatenated = ReshapeFromHeads(attnOutput, _numQueryHeads, seqLen, kvHeadDim);

        // Output projection
        var output = TensorOperations.MatMul(concatenated, Transpose2D(oProjWeight));

        return output;
    }

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
    /// Computes Q @ K^T / sqrt(dotDim) for each head independently.
    /// When Q and K have different per-head dimensions, the dot product
    /// is computed over the smaller dimension (kvHeadDim).
    /// </summary>
    private static Tensor.Tensor ComputeAttentionScores(
        Tensor.Tensor Q, Tensor.Tensor K, int numHeads, int queryLen, int kvLen, int qHeadDim, int kvHeadDim)
    {
        var dotDim = Math.Min(qHeadDim, kvHeadDim);
        var scores = new float[numHeads * queryLen * kvLen];
        var scale = 1.0f / MathF.Sqrt(dotDim);

        for (var h = 0; h < numHeads; h++)
        {
            var qOffset = h * queryLen * qHeadDim;
            var kOffset = h * kvLen * kvHeadDim;
            var sOffset = h * queryLen * kvLen;

            for (var i = 0; i < queryLen; i++)
            {
                for (var j = 0; j < kvLen; j++)
                {
                    var dot = 0.0f;

                    for (var d = 0; d < dotDim; d++)
                    {
                        dot += Q.Data[qOffset + i * qHeadDim + d] * K.Data[kOffset + j * kvHeadDim + d];
                    }

                    scores[sOffset + i * kvLen + j] = dot * scale;
                }
            }
        }

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
                        scores[h, i, j] = float.NegativeInfinity;
                        continue;
                    }

                    // Sliding window mask (only for sliding attention layers)
                    if (!_isFullAttention && queryPos - j >= _slidingWindowSize)
                    {
                        scores[h, i, j] = float.NegativeInfinity;
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
