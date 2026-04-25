using System;
using System.Collections.Generic;
using System.Numerics;
using System.Threading.Tasks;

namespace WebExpress.LLM.Tensor;

/// <summary>
/// Provides static methods for common tensor operations used in transformer inference,
/// including matrix multiplication, softmax, GELU activation, and RMS normalization.
/// </summary>
/// <remarks>
/// All operations are implemented using native .NET functionality. No external numerical
/// libraries are required.
/// </remarks>
public static class TensorOperations
{
    private const int TileSize = 64;

    /// <summary>
    /// Performs matrix multiplication between two 2D tensors.
    /// </summary>
    /// <param name="a">Left operand with shape [M, K].</param>
    /// <param name="b">Right operand with shape [K, N].</param>
    /// <returns>A new tensor with shape [M, N].</returns>
    /// <exception cref="ArgumentException">Thrown when inner dimensions do not match.</exception>
    public static Tensor MatMul(Tensor a, Tensor b)
    {
        ArgumentNullException.ThrowIfNull(a);
        ArgumentNullException.ThrowIfNull(b);

        if (a.Rank != 2 || b.Rank != 2)
        {
            throw new ArgumentException("MatMul requires 2D tensors.");
        }

        var m = a.Shape[0];
        var k = a.Shape[1];
        var n = b.Shape[1];

        if (k != b.Shape[0])
        {
            throw new ArgumentException(
                $"Inner dimensions must match for MatMul: [{m},{k}] x [{b.Shape[0]},{n}].");
        }

        var result = new float[m * n];
        var vecWidth = Vector<float>.Count;

        Parallel.For(0, m, i =>
        {
            var aRowOffset = i * k;
            var rRowOffset = i * n;

            for (var p = 0; p < k; p++)
            {
                var aVal = a.Data[aRowOffset + p];
                if (aVal == 0f)
                    continue;

                var aVec = new Vector<float>(aVal);
                var bRowOffset = p * n;

                var j = 0;
                for (; j <= n - vecWidth; j += vecWidth)
                {
                    var bVec = new Vector<float>(b.Data, bRowOffset + j);
                    var rVec = new Vector<float>(result, rRowOffset + j);

                    rVec += aVec * bVec;
                    rVec.CopyTo(result, rRowOffset + j);
                }
                for (; j < n; j++)
                {
                    result[rRowOffset + j] += aVal * b.Data[bRowOffset + j];
                }
            }
        });

        return new Tensor([m, n], result);
    }

    /// <summary>
    /// Applies the softmax function along the last dimension of the tensor.
    /// </summary>
    /// <param name="input">The input tensor.</param>
    /// <returns>A new tensor with softmax applied along the last dimension.</returns>
    public static Tensor Softmax(Tensor input)
    {
        ArgumentNullException.ThrowIfNull(input);

        var data = input.Data;
        var result = new float[data.Length];
        var lastDim = input.Shape[^1];
        var outerSize = data.Length / lastDim;

        for (var outer = 0; outer < outerSize; outer++)
        {
            var offset = outer * lastDim;

            // Find max for numerical stability
            var max = float.NegativeInfinity;

            for (var i = 0; i < lastDim; i++)
            {
                if (data[offset + i] > max)
                {
                    max = data[offset + i];
                }
            }

            // Compute exp and sum
            var sum = 0.0f;

            for (var i = 0; i < lastDim; i++)
            {
                result[offset + i] = MathF.Exp(data[offset + i] - max);
                sum += result[offset + i];
            }

            // Normalize
            if (sum > 0)
            {
                for (var i = 0; i < lastDim; i++)
                {
                    result[offset + i] /= sum;
                }
            }
        }

        return new Tensor(ToIntArray(input.Shape), result);
    }

    /// <summary>
    /// Applies the GELU (Gaussian Error Linear Unit) activation function element-wise.
    /// Uses the approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    /// </summary>
    /// <param name="input">The input tensor.</param>
    /// <returns>A new tensor with GELU applied element-wise.</returns>
    public static Tensor Gelu(Tensor input)
    {
        ArgumentNullException.ThrowIfNull(input);

        var data = input.Data;
        var result = new float[data.Length];
        var sqrt2OverPi = MathF.Sqrt(2.0f / MathF.PI);

        for (var i = 0; i < data.Length; i++)
        {
            var x = data[i];
            var inner = sqrt2OverPi * (x + 0.044715f * x * x * x);
            result[i] = 0.5f * x * (1.0f + MathF.Tanh(inner));
        }

        return new Tensor(ToIntArray(input.Shape), result);
    }

    /// <summary>
    /// Applies RMS (Root Mean Square) normalization along the last dimension.
    /// </summary>
    /// <remarks>
    /// Applies Root Mean Square Layer Normalization (RMSNorm) across the last dimension of the input tensor.
    /// For each vector along the last axis, computes the root mean square, normalizes, and scales by a learned weight.
    /// Formula:
    ///   y_i = (x_i / sqrt(mean_j(x_j^2) + epsilon)) * weight_i
    /// Used in transformer models such as Gemma and Llama as a stable normalization technique.
    /// </remarks>
    /// <param name="input">The input tensor.</param>
    /// <param name="weight">The normalization weight tensor (1D, same size as last dimension). Pass <c>null</c> for a scale-less RMSNorm, as used by Gemma-4 <c>value_norm</c> and the MoE <c>router_norm</c>.</param>
    /// <param name="epsilon">Small constant for numerical stability.</param>
    /// <returns>A new normalized tensor.</returns>
    public static Tensor RmsNorm(Tensor input, Tensor weight, float epsilon = 1e-6f)
    {
        ArgumentNullException.ThrowIfNull(input);

        var data = input.Data;
        var lastDim = input.Shape[^1];
        var wData = weight?.Data;

        if (wData != null && wData.Length != lastDim)
        {
            throw new ArgumentException(
                $"Weight dimension {wData.Length} does not match last input dimension {lastDim}.");
        }

        var result = new float[data.Length];
        var outerSize = data.Length / lastDim;

        Parallel.For(0, outerSize, outer =>
        {
            var offset = outer * lastDim;

            var sumSquares = 0.0f;

            for (var i = 0; i < lastDim; i++)
            {
                sumSquares += data[offset + i] * data[offset + i];
            }

            var rms = MathF.Sqrt(sumSquares / lastDim + epsilon);

            if (wData == null)
            {
                for (var i = 0; i < lastDim; i++)
                {
                    result[offset + i] = data[offset + i] / rms;
                }
            }
            else
            {
                for (var i = 0; i < lastDim; i++)
                {
                    result[offset + i] = data[offset + i] / rms * wData[i];
                }
            }
        });

        return new Tensor(ToIntArray(input.Shape), result);
    }

    /// <summary>
    /// Applies the hyperbolic tangent function element-wise.
    /// </summary>
    public static Tensor Tanh(Tensor input)
    {
        ArgumentNullException.ThrowIfNull(input);

        var data = input.Data;
        var result = new float[data.Length];

        for (var i = 0; i < data.Length; i++)
        {
            result[i] = MathF.Tanh(data[i]);
        }

        return new Tensor(ToIntArray(input.Shape), result);
    }

    /// <summary>
    /// Computes element-wise square root.
    /// </summary>
    public static Tensor Sqrt(Tensor input)
    {
        ArgumentNullException.ThrowIfNull(input);

        var data = input.Data;
        var result = new float[data.Length];

        for (var i = 0; i < data.Length; i++)
        {
            result[i] = MathF.Sqrt(data[i]);
        }

        return new Tensor(ToIntArray(input.Shape), result);
    }

    /// <summary>
    /// Concatenates two tensors along the specified dimension.
    /// </summary>
    /// <param name="a">First tensor.</param>
    /// <param name="b">Second tensor.</param>
    /// <param name="dim">The dimension along which to concatenate.</param>
    /// <returns>A new tensor that is the concatenation of a and b along the specified dimension.</returns>
    public static Tensor Concatenate(Tensor a, Tensor b, int dim)
    {
        ArgumentNullException.ThrowIfNull(a);
        ArgumentNullException.ThrowIfNull(b);

        if (a.Rank != b.Rank)
        {
            throw new ArgumentException("Tensors must have the same number of dimensions.");
        }

        if (dim < 0 || dim >= a.Rank)
        {
            throw new ArgumentOutOfRangeException(nameof(dim));
        }

        for (var i = 0; i < a.Rank; i++)
        {
            if (i != dim && a.Shape[i] != b.Shape[i])
            {
                throw new ArgumentException($"All dimensions except dim {dim} must match.");
            }
        }

        var newShape = new int[a.Rank];

        for (var i = 0; i < a.Rank; i++)
        {
            newShape[i] = i == dim ? a.Shape[i] + b.Shape[i] : a.Shape[i];
        }

        var result = new Tensor(newShape);

        // Compute strides for copy
        var outerSize = 1;

        for (var i = 0; i < dim; i++)
        {
            outerSize *= newShape[i];
        }

        var aInnerSize = a.Shape[dim];
        var bInnerSize = b.Shape[dim];

        var trailingSize = 1;

        for (var i = dim + 1; i < a.Rank; i++)
        {
            trailingSize *= newShape[i];
        }

        for (var outer = 0; outer < outerSize; outer++)
        {
            var aStart = outer * aInnerSize * trailingSize;
            var bStart = outer * bInnerSize * trailingSize;
            var rStart = outer * (aInnerSize + bInnerSize) * trailingSize;

            Array.Copy(a.Data, aStart, result.Data, rStart, aInnerSize * trailingSize);
            Array.Copy(b.Data, bStart, result.Data, rStart + aInnerSize * trailingSize, bInnerSize * trailingSize);
        }

        return result;
    }

    /// <summary>
    /// Creates a causal attention mask (upper triangular with -infinity).
    /// </summary>
    /// <param name="seqLen">The sequence length.</param>
    /// <returns>A [seqLen, seqLen] tensor where future positions are masked with -infinity.</returns>
    public static Tensor CausalMask(int seqLen)
    {
        var data = new float[seqLen * seqLen];

        for (var i = 0; i < seqLen; i++)
        {
            for (var j = 0; j < seqLen; j++)
            {
                data[i * seqLen + j] = j <= i ? 0.0f : float.NegativeInfinity;
            }
        }

        return new Tensor([seqLen, seqLen], data);
    }

    /// <summary>
    /// Creates a sliding window attention mask. Positions outside the window and future positions
    /// are masked with -infinity.
    /// </summary>
    /// <param name="seqLen">The sequence length.</param>
    /// <param name="windowSize">The sliding window size.</param>
    /// <returns>A [seqLen, seqLen] mask tensor.</returns>
    public static Tensor SlidingWindowMask(int seqLen, int windowSize)
    {
        var data = new float[seqLen * seqLen];

        for (var i = 0; i < seqLen; i++)
        {
            for (var j = 0; j < seqLen; j++)
            {
                // Allow positions: j <= i (causal) AND j >= i - windowSize + 1 (within window)
                var isCausal = j <= i;
                var isInWindow = j >= i - windowSize + 1;
                data[i * seqLen + j] = isCausal && isInWindow ? 0.0f : float.NegativeInfinity;
            }
        }

        return new Tensor([seqLen, seqLen], data);
    }

    /// <summary>
    /// Applies the SiLU (Sigmoid Linear Unit) activation function element-wise.
    /// SiLU(x) = x * sigmoid(x)
    /// </summary>
    public static Tensor Silu(Tensor input)
    {
        ArgumentNullException.ThrowIfNull(input);

        var data = input.Data;
        var result = new float[data.Length];

        for (var i = 0; i < data.Length; i++)
        {
            var x = data[i];
            result[i] = x / (1.0f + MathF.Exp(-x));
        }

        return new Tensor(ToIntArray(input.Shape), result);
    }

    /// <summary>
    /// Looks up embedding vectors for the given token IDs from an embedding weight matrix.
    /// </summary>
    /// <remarks>
    /// Looks up the embedding vector for each token in the input sequence.
    /// For each token ID, copies the corresponding row from the embedding weight matrix
    /// ([vocab_size, hidden_size]) to the result tensor ([seq_len, hidden_size]).
    /// This operation is typically used as the first step in a transformer model's forward pass.
    /// Throws an exception if a token ID is out of range.
    /// </remarks>
    /// <param name="embeddingWeights">The embedding weight matrix [vocab_size, hidden_size].</param>
    /// <param name="tokenIds">The token IDs to look up.</param>
    /// <returns>A [seq_len, hidden_size] tensor of embedded tokens.</returns>
    public static Tensor EmbeddingLookup(Tensor embeddingWeights, int[] tokenIds)
    {
        ArgumentNullException.ThrowIfNull(embeddingWeights);
        ArgumentNullException.ThrowIfNull(tokenIds);

        if (embeddingWeights.Rank != 2)
        {
            throw new ArgumentException("Embedding weights must be a 2D tensor [vocab_size, hidden_size].");
        }

        var hiddenSize = embeddingWeights.Shape[1];
        var seqLen = tokenIds.Length;
        var result = new float[seqLen * hiddenSize];

        var vocabSize = embeddingWeights.Shape[0];

        for (var i = 0; i < seqLen; i++)
        {
            var tokenId = tokenIds[i];

            if (tokenId < 0 || tokenId >= vocabSize)
            {
                throw new ArgumentOutOfRangeException(nameof(tokenIds),
                    $"Token ID {tokenId} at position {i} is out of range for vocabulary size {vocabSize}.");
            }

            Array.Copy(embeddingWeights.Data, tokenId * hiddenSize, result, i * hiddenSize, hiddenSize);
        }

        return new Tensor([seqLen, hiddenSize], result);
    }

    /// <summary>
    /// Converts the specified read-only list of integers to a new array.
    /// </summary>
    /// <param name="shape">The read-only list of integers to convert. Cannot be null.</param>
    /// <returns>An array containing the elements of the specified list, in the same order.</returns>
    private static int[] ToIntArray(IReadOnlyList<int> shape)
    {
        var arr = new int[shape.Count];

        for (var i = 0; i < shape.Count; i++)
        {
            arr[i] = shape[i];
        }

        return arr;
    }
}
