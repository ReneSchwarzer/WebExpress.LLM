using System;
using WebExpress.LLM.Tensor;

namespace WebExpress.LLM.Gemma;

/// <summary>
/// Implements rotary position embeddings (RoPE) for encoding sequence order
/// in transformer attention mechanisms.
/// </summary>
/// <remarks>
/// RoPE encodes position information by rotating pairs of dimensions in the
/// query and key vectors. Gemma-4 uses two RoPE configurations:
/// - Sliding attention: default RoPE with theta=10000
/// - Full attention: proportional RoPE with theta=1000000 and partial rotation (25%)
/// </remarks>
public sealed class RotaryEmbedding
{
    private readonly float _theta;
    private readonly float _partialRotaryFactor;

    /// <summary>
    /// Initializes a new RotaryEmbedding with the specified parameters.
    /// </summary>
    /// <param name="theta">The base frequency for the rotary embeddings.</param>
    /// <param name="partialRotaryFactor">Fraction of head dimension to rotate (1.0 = full, 0.25 = 25%).</param>
    public RotaryEmbedding(float theta = 10000.0f, float partialRotaryFactor = 1.0f)
    {
        if (theta <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(theta), "Theta must be positive.");
        }

        if (partialRotaryFactor <= 0 || partialRotaryFactor > 1.0f)
        {
            throw new ArgumentOutOfRangeException(nameof(partialRotaryFactor), "Partial rotary factor must be in (0, 1].");
        }

        _theta = theta;
        _partialRotaryFactor = partialRotaryFactor;
    }

    /// <summary>
    /// Applies rotary position embeddings to the input tensor.
    /// </summary>
    /// <param name="input">Input tensor of shape [seqLen, headDim] or [numHeads, seqLen, headDim].</param>
    /// <param name="startPosition">The starting position index (for KV cache continuation).</param>
    /// <returns>A new tensor with rotary embeddings applied.</returns>
    public Tensor.Tensor Apply(Tensor.Tensor input, int startPosition = 0)
    {
        ArgumentNullException.ThrowIfNull(input);

        if (input.Rank == 2)
        {
            return Apply2D(input, startPosition);
        }

        if (input.Rank == 3)
        {
            return Apply3D(input, startPosition);
        }

        throw new ArgumentException("RoPE requires 2D [seqLen, headDim] or 3D [numHeads, seqLen, headDim] tensor.");
    }

    private Tensor.Tensor Apply2D(Tensor.Tensor input, int startPosition)
    {
        var seqLen = input.Shape[0];
        var headDim = input.Shape[1];
        var rotDim = (int)(headDim * _partialRotaryFactor);

        // Ensure rotDim is even
        rotDim = rotDim / 2 * 2;

        var result = input.Clone();

        for (var pos = 0; pos < seqLen; pos++)
        {
            var position = pos + startPosition;

            for (var i = 0; i < rotDim; i += 2)
            {
                var freq = 1.0f / MathF.Pow(_theta, (float)i / rotDim);
                var angle = position * freq;
                var cos = MathF.Cos(angle);
                var sin = MathF.Sin(angle);

                var x0 = input[pos, i];
                var x1 = input[pos, i + 1];

                result[pos, i] = x0 * cos - x1 * sin;
                result[pos, i + 1] = x0 * sin + x1 * cos;
            }
        }

        return result;
    }

    private Tensor.Tensor Apply3D(Tensor.Tensor input, int startPosition)
    {
        var numHeads = input.Shape[0];
        var seqLen = input.Shape[1];
        var headDim = input.Shape[2];
        var rotDim = (int)(headDim * _partialRotaryFactor);
        rotDim = rotDim / 2 * 2;

        var result = input.Clone();

        for (var h = 0; h < numHeads; h++)
        {
            for (var pos = 0; pos < seqLen; pos++)
            {
                var position = pos + startPosition;

                for (var i = 0; i < rotDim; i += 2)
                {
                    var freq = 1.0f / MathF.Pow(_theta, (float)i / rotDim);
                    var angle = position * freq;
                    var cos = MathF.Cos(angle);
                    var sin = MathF.Sin(angle);

                    var x0 = input[h, pos, i];
                    var x1 = input[h, pos, i + 1];

                    result[h, pos, i] = x0 * cos - x1 * sin;
                    result[h, pos, i + 1] = x0 * sin + x1 * cos;
                }
            }
        }

        return result;
    }
}
