using System;
using System.Threading.Tasks;

namespace WebExpress.LLM.Gemma;

/// <summary>
/// Implements rotary position embeddings (RoPE) for encoding sequence order
/// in transformer attention mechanisms.
/// </summary>
/// <remarks>
/// Uses the split-halves rotation pattern from the Google DeepMind Gemma-4
/// reference (<c>gemma/gm/math/_positional_embeddings.apply_rope</c>):
/// the head dimension is split into two halves and the rotation pairs are
/// <c>(x_i, x_{i + H/2})</c>, not consecutive <c>(x_{2i}, x_{2i+1})</c>.
/// The frequency denominator is always the full <c>head_dim</c>, so partial
/// rotary factors only affect how many angles are rotated — not the angles
/// themselves. Dimensions beyond the rotated range are left unchanged.
///
/// Gemma-4 uses two RoPE configurations:
/// - Sliding (local) attention: theta = 10 000, rope_proportion = 1.0
/// - Full    (global) attention: theta = 1 000 000, rope_proportion = 0.25
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

    /// <summary>
    /// Computes <c>rope_angles = int(partial_rotary_factor * head_dim / 2)</c> — the
    /// number of angle pairs to rotate. The rotated dimension count is
    /// <c>2 * rope_angles</c>; indices <c>[rope_angles, head_dim/2)</c> of the first
    /// half and <c>[head_dim/2 + rope_angles, head_dim)</c> of the second half are
    /// left unchanged ("nope" — no positional embedding).
    /// </summary>
    private static int ComputeRopeAngles(int headDim, float partialRotaryFactor)
    {
        return (int)(partialRotaryFactor * headDim / 2);
    }

    /// <summary>
    /// Applies RoPE to a 2D tensor of shape [seqLen, headDim].
    /// </summary>
    /// <remarks>
    /// Uses the split-halves rotation pattern:
    ///   out[pos, j]             = x[pos, j]             * cos - x[pos, j + H/2] * sin
    ///   out[pos, j + H/2]       = x[pos, j + H/2]       * cos + x[pos, j]       * sin
    /// for j in [0, rope_angles). Indices outside that range copy through unchanged.
    /// </remarks>
    private Tensor.Tensor Apply2D(Tensor.Tensor input, int startPosition)
    {
        var seqLen = input.Shape[0];
        var headDim = input.Shape[1];
        var half = headDim / 2;
        var ropeAngles = ComputeRopeAngles(headDim, _partialRotaryFactor);

        var result = input.Clone();

        for (var pos = 0; pos < seqLen; pos++)
        {
            var position = pos + startPosition;

            for (var j = 0; j < ropeAngles; j++)
            {
                var freq = 1.0f / MathF.Pow(_theta, 2.0f * j / headDim);
                var angle = position * freq;
                var cos = MathF.Cos(angle);
                var sin = MathF.Sin(angle);

                var x0 = input[pos, j];
                var x1 = input[pos, j + half];

                result[pos, j] = x0 * cos - x1 * sin;
                result[pos, j + half] = x1 * cos + x0 * sin;
            }
        }

        return result;
    }

    /// <summary>
    /// Applies RoPE to a 3D tensor of shape [numHeads, seqLen, headDim] using the
    /// same split-halves pattern as <see cref="Apply2D"/>.
    /// </summary>
    private Tensor.Tensor Apply3D(Tensor.Tensor input, int startPosition)
    {
        var numHeads = input.Shape[0];
        var seqLen = input.Shape[1];
        var headDim = input.Shape[2];
        var half = headDim / 2;
        var ropeAngles = ComputeRopeAngles(headDim, _partialRotaryFactor);

        var result = input.Clone();

        Parallel.For(0, numHeads, h =>
        {
            for (var pos = 0; pos < seqLen; pos++)
            {
                var position = pos + startPosition;

                for (var j = 0; j < ropeAngles; j++)
                {
                    var freq = 1.0f / MathF.Pow(_theta, 2.0f * j / headDim);
                    var angle = position * freq;
                    var cos = MathF.Cos(angle);
                    var sin = MathF.Sin(angle);

                    var x0 = input[h, pos, j];
                    var x1 = input[h, pos, j + half];

                    result[h, pos, j] = x0 * cos - x1 * sin;
                    result[h, pos, j + half] = x1 * cos + x0 * sin;
                }
            }
        });

        return result;
    }
}
