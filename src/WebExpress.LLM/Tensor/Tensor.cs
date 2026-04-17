using System;
using System.Collections.Generic;
using System.Linq;

namespace WebExpress.LLM.Tensor;

/// <summary>
/// A multi-dimensional array of single-precision floating-point values that provides
/// the numerical foundation for transformer inference operations.
/// </summary>
/// <remarks>
/// This tensor implementation is designed for CPU-based inference and supports
/// 1D, 2D, and 3D shapes commonly used in transformer models. All operations
/// are implemented using native .NET functionality without external dependencies.
/// </remarks>
public sealed class Tensor
{
    private readonly float[] _data;
    private readonly int[] _shape;

    /// <summary>
    /// Initializes a new tensor with the specified shape, filled with zeros.
    /// </summary>
    /// <param name="shape">The dimensions of the tensor.</param>
    /// <exception cref="ArgumentException">Thrown when shape is null, empty, or contains non-positive dimensions.</exception>
    public Tensor(params int[] shape)
    {
        ValidateShape(shape);

        _shape = (int[])shape.Clone();
        _data = new float[ComputeLength(shape)];
    }

    /// <summary>
    /// Initializes a new tensor with the specified shape and data.
    /// </summary>
    /// <param name="shape">The dimensions of the tensor.</param>
    /// <param name="data">The data to fill the tensor with. Must match the total element count implied by shape.</param>
    /// <exception cref="ArgumentException">Thrown when shape is invalid or data length does not match shape.</exception>
    public Tensor(int[] shape, float[] data)
    {
        ValidateShape(shape);
        ArgumentNullException.ThrowIfNull(data);

        var expectedLength = ComputeLength(shape);

        if (data.Length != expectedLength)
        {
            throw new ArgumentException(
                $"Data length {data.Length} does not match shape {string.Join("x", shape)} (expected {expectedLength}).");
        }

        _shape = (int[])shape.Clone();
        _data = (float[])data.Clone();
    }

    /// <summary>
    /// Initializes a new tensor with the specified shape, wrapping the provided data without copying.
    /// </summary>
    private Tensor(int[] shape, float[] data, bool noCopy)
    {
        _shape = shape;
        _data = data;
    }

    /// <summary>
    /// Gets the shape of this tensor as a read-only list of dimension sizes.
    /// </summary>
    public IReadOnlyList<int> Shape => _shape;

    /// <summary>
    /// Gets the total number of elements in this tensor.
    /// </summary>
    public int Length => _data.Length;

    /// <summary>
    /// Gets the number of dimensions (rank) of this tensor.
    /// </summary>
    public int Rank => _shape.Length;

    /// <summary>
    /// Gets the underlying data array. For performance-critical internal use only.
    /// </summary>
    internal float[] Data => _data;

    /// <summary>
    /// Gets or sets the element at the specified flat index.
    /// </summary>
    public float this[int index]
    {
        get => _data[index];
        set => _data[index] = value;
    }

    /// <summary>
    /// Gets or sets the element at the specified 2D coordinates.
    /// </summary>
    public float this[int row, int col]
    {
        get => _data[row * _shape[^1] + col];
        set => _data[row * _shape[^1] + col] = value;
    }

    /// <summary>
    /// Gets or sets the element at the specified 3D coordinates.
    /// </summary>
    public float this[int dim0, int dim1, int dim2]
    {
        get => _data[(dim0 * _shape[1] + dim1) * _shape[2] + dim2];
        set => _data[(dim0 * _shape[1] + dim1) * _shape[2] + dim2] = value;
    }

    /// <summary>
    /// Creates a tensor filled with zeros.
    /// </summary>
    public static Tensor Zeros(params int[] shape)
    {
        return new Tensor(shape);
    }

    /// <summary>
    /// Creates a tensor filled with ones.
    /// </summary>
    public static Tensor Ones(params int[] shape)
    {
        var t = new Tensor(shape);
        Array.Fill(t._data, 1.0f);
        return t;
    }

    /// <summary>
    /// Creates a 1D tensor from the specified values.
    /// </summary>
    public static Tensor FromArray(float[] values)
    {
        ArgumentNullException.ThrowIfNull(values);
        return new Tensor([values.Length], values);
    }

    /// <summary>
    /// Creates a 2D tensor from a jagged array.
    /// </summary>
    public static Tensor From2DArray(float[][] values)
    {
        ArgumentNullException.ThrowIfNull(values);

        if (values.Length == 0)
        {
            throw new ArgumentException("Values array must not be empty.");
        }

        var rows = values.Length;
        var cols = values[0].Length;
        var data = new float[rows * cols];

        for (var i = 0; i < rows; i++)
        {
            if (values[i].Length != cols)
            {
                throw new ArgumentException("All rows must have the same length.");
            }

            Array.Copy(values[i], 0, data, i * cols, cols);
        }

        return new Tensor([rows, cols], data, noCopy: true);
    }

    /// <summary>
    /// Returns a copy of the underlying data as a flat array.
    /// </summary>
    public float[] ToArray()
    {
        return (float[])_data.Clone();
    }

    /// <summary>
    /// Extracts a single row from a 2D tensor or a single slice along the first dimension.
    /// </summary>
    /// <param name="index">The index along the first dimension.</param>
    /// <returns>A new tensor representing the selected slice.</returns>
    public Tensor GetRow(int index)
    {
        if (_shape.Length < 2)
        {
            throw new InvalidOperationException("GetRow requires at least a 2D tensor.");
        }

        if (index < 0 || index >= _shape[0])
        {
            throw new ArgumentOutOfRangeException(nameof(index),
                $"Index {index} is out of range for first dimension of size {_shape[0]}.");
        }

        var innerSize = 1;

        for (var i = 1; i < _shape.Length; i++)
        {
            innerSize *= _shape[i];
        }

        var newShape = new int[_shape.Length - 1];
        Array.Copy(_shape, 1, newShape, 0, newShape.Length);

        var newData = new float[innerSize];
        Array.Copy(_data, index * innerSize, newData, 0, innerSize);

        return new Tensor(newShape, newData, noCopy: true);
    }

    /// <summary>
    /// Extracts the last row/slice along the first dimension.
    /// Commonly used to get logits for the last position in autoregressive generation.
    /// </summary>
    public Tensor GetLastRow()
    {
        return GetRow(_shape[0] - 1);
    }

    /// <summary>
    /// Creates a reshaped view of this tensor. The total number of elements must remain the same.
    /// </summary>
    public Tensor Reshape(params int[] newShape)
    {
        var newLength = ComputeLength(newShape);

        if (newLength != _data.Length)
        {
            throw new ArgumentException(
                $"Cannot reshape tensor of length {_data.Length} into shape {string.Join("x", newShape)} (length {newLength}).");
        }

        return new Tensor((int[])newShape.Clone(), _data, noCopy: true);
    }

    /// <summary>
    /// Transposes a 2D tensor (swaps rows and columns).
    /// </summary>
    public Tensor Transpose()
    {
        if (_shape.Length != 2)
        {
            throw new InvalidOperationException("Transpose is only supported for 2D tensors.");
        }

        var rows = _shape[0];
        var cols = _shape[1];
        var result = new float[rows * cols];

        for (var i = 0; i < rows; i++)
        {
            for (var j = 0; j < cols; j++)
            {
                result[j * rows + i] = _data[i * cols + j];
            }
        }

        return new Tensor([cols, rows], result, noCopy: true);
    }

    /// <summary>
    /// Returns a deep copy of this tensor.
    /// </summary>
    public Tensor Clone()
    {
        return new Tensor((int[])_shape.Clone(), (float[])_data.Clone(), noCopy: true);
    }

    // ---------------------------------------------------------------
    // Arithmetic operators
    // ---------------------------------------------------------------

    /// <summary>
    /// Element-wise addition of two tensors with broadcasting support.
    /// </summary>
    public static Tensor operator +(Tensor a, Tensor b)
    {
        return ElementWise(a, b, static (x, y) => x + y);
    }

    /// <summary>
    /// Element-wise subtraction of two tensors with broadcasting support.
    /// </summary>
    public static Tensor operator -(Tensor a, Tensor b)
    {
        return ElementWise(a, b, static (x, y) => x - y);
    }

    /// <summary>
    /// Element-wise multiplication of two tensors with broadcasting support.
    /// </summary>
    public static Tensor operator *(Tensor a, Tensor b)
    {
        return ElementWise(a, b, static (x, y) => x * y);
    }

    /// <summary>
    /// Scalar multiplication.
    /// </summary>
    public static Tensor operator *(Tensor a, float scalar)
    {
        var result = new float[a._data.Length];

        for (var i = 0; i < result.Length; i++)
        {
            result[i] = a._data[i] * scalar;
        }

        return new Tensor((int[])a._shape.Clone(), result, noCopy: true);
    }

    /// <summary>
    /// Scalar multiplication (commutative).
    /// </summary>
    public static Tensor operator *(float scalar, Tensor a)
    {
        return a * scalar;
    }

    /// <summary>
    /// Scalar addition.
    /// </summary>
    public static Tensor operator +(Tensor a, float scalar)
    {
        var result = new float[a._data.Length];

        for (var i = 0; i < result.Length; i++)
        {
            result[i] = a._data[i] + scalar;
        }

        return new Tensor((int[])a._shape.Clone(), result, noCopy: true);
    }

    /// <summary>
    /// Element-wise division of two tensors with broadcasting support.
    /// </summary>
    public static Tensor operator /(Tensor a, Tensor b)
    {
        return ElementWise(a, b, static (x, y) => x / y);
    }

    /// <summary>
    /// Scalar division.
    /// </summary>
    public static Tensor operator /(Tensor a, float scalar)
    {
        var result = new float[a._data.Length];

        for (var i = 0; i < result.Length; i++)
        {
            result[i] = a._data[i] / scalar;
        }

        return new Tensor((int[])a._shape.Clone(), result, noCopy: true);
    }

    /// <summary>
    /// Negation.
    /// </summary>
    public static Tensor operator -(Tensor a)
    {
        return a * -1.0f;
    }

    /// <summary>
    /// Validates that the specified shape array is non-null, non-empty, and contains only positive integers.
    /// </summary>
    /// <param name="shape">
    /// An array of integers representing the dimensions of a shape. Each element must be a positive integer.
    /// </param>
    /// <exception cref="ArgumentException">
    /// Thrown if the shape array is null, empty, or contains a non-positive value.
    /// </exception>
    private static void ValidateShape(int[] shape)
    {
        if (shape == null || shape.Length == 0)
        {
            throw new ArgumentException("Shape must be a non-empty array of positive integers.");
        }

        for (var i = 0; i < shape.Length; i++)
        {
            if (shape[i] <= 0)
            {
                throw new ArgumentException($"Shape dimension {i} must be positive, but was {shape[i]}.");
            }
        }
    }

    /// <summary>
    /// Calculates the total number of elements represented by the specified shape dimensions.
    /// </summary>
    /// <param name="shape">
    /// An array of integers specifying the size of each dimension. Each value must be greater than or equal to zero.
    /// </param>
    /// <returns>
    /// The product of all dimension sizes in the shape array. Returns 1 if the array is empty.
    /// </returns>
    private static int ComputeLength(int[] shape)
    {
        var length = 1;

        for (var i = 0; i < shape.Length; i++)
        {
            length *= shape[i];
        }

        return length;
    }

    /// <summary>
    /// Performs element-wise operation between two tensors with broadcasting.
    /// Supports broadcasting when shapes differ (e.g. [3,4] op [1,4] or [3,4] op [4]).
    /// </summary>
    private static Tensor ElementWise(Tensor a, Tensor b, Func<float, float, float> op)
    {
        // Fast path: identical shapes
        if (a._shape.SequenceEqual(b._shape))
        {
            var result = new float[a._data.Length];

            for (var i = 0; i < result.Length; i++)
            {
                result[i] = op(a._data[i], b._data[i]);
            }

            return new Tensor((int[])a._shape.Clone(), result, noCopy: true);
        }

        // Broadcasting path
        var outShape = BroadcastShapes(a._shape, b._shape);
        var outLength = ComputeLength(outShape);
        var outData = new float[outLength];
        var outRank = outShape.Length;

        var aStrides = ComputeBroadcastStrides(a._shape, outShape);
        var bStrides = ComputeBroadcastStrides(b._shape, outShape);

        var indices = new int[outRank];

        for (var i = 0; i < outLength; i++)
        {
            var aIndex = 0;
            var bIndex = 0;

            for (var d = 0; d < outRank; d++)
            {
                aIndex += indices[d] * aStrides[d];
                bIndex += indices[d] * bStrides[d];
            }

            outData[i] = op(a._data[aIndex], b._data[bIndex]);

            // Increment multi-dimensional index
            for (var d = outRank - 1; d >= 0; d--)
            {
                indices[d]++;

                if (indices[d] < outShape[d])
                {
                    break;
                }

                indices[d] = 0;
            }
        }

        return new Tensor(outShape, outData, noCopy: true);
    }

    /// <summary>
    /// Determines the broadcast‑compatible shape of two arrays by combining their dimensions
    /// according to standard broadcasting rules.
    /// </summary>
    /// <remarks>
    /// The method follows the broadcasting rules used in numerical libraries such as NumPy.
    /// Dimensions are compared from right to left; a dimension is considered compatible if it is
    /// equal or if one of the values is 1.
    /// </remarks>
    /// <param name="a">
    /// The first array of dimensions describing the shape of a tensor. All values must be greater than 0.
    /// </param>
    /// <param name="b">
    /// The second array of dimensions describing the shape of a tensor. All values must be greater than 0.
    /// </param>
    /// <returns>
    /// An array of integers representing the broadcasted shape of both inputs. Each dimension corresponds
    /// to the maximum compatible size at that position.
    /// </returns>
    /// <exception cref="InvalidOperationException">
    /// Thrown when the input shapes are not broadcast‑compatible.
    /// </exception>
    private static int[] BroadcastShapes(int[] a, int[] b)
    {
        var maxRank = Math.Max(a.Length, b.Length);
        var result = new int[maxRank];

        for (var i = 0; i < maxRank; i++)
        {
            var dimA = i < maxRank - a.Length ? 1 : a[i - (maxRank - a.Length)];
            var dimB = i < maxRank - b.Length ? 1 : b[i - (maxRank - b.Length)];

            if (dimA != dimB && dimA != 1 && dimB != 1)
            {
                throw new InvalidOperationException(
                    $"Cannot broadcast shapes [{string.Join(",", a)}] and [{string.Join(",", b)}].");
            }

            result[i] = Math.Max(dimA, dimB);
        }

        return result;
    }

    /// <summary>
    /// Computes the strides for an input array relative to an output shape in order to support broadcasting.
    /// </summary>
    /// <remarks>
    /// The returned strides can be used to map indices from the broadcasted shape back to the
    /// original input array. This is particularly useful when implementing array operations
    /// that rely on broadcasting semantics.
    /// </remarks>
    /// <param name="shape">
    /// The shape of the input array. Each element specifies the size of the corresponding
    /// dimension. The array must contain at least as many dimensions as required for broadcasting.
    /// </param>
    /// <param name="outShape">
    /// The desired shape after broadcasting. Each element specifies the size of the corresponding
    /// dimension in the broadcasted representation.
    /// </param>
    /// <returns>
    /// An array of integers representing the stride for each dimension in the broadcasted shape.
    /// A value of 0 indicates that the corresponding dimension is broadcasted.
    /// </returns>
    private static int[] ComputeBroadcastStrides(int[] shape, int[] outShape)
    {
        // Compute strides for shape relative to outShape for broadcasting
        var rank = outShape.Length;
        var strides = new int[rank];
        var offset = rank - shape.Length;

        // Compute the actual strides in the original array
        var originalStrides = new int[shape.Length];

        if (shape.Length > 0)
        {
            originalStrides[^1] = 1;

            for (var i = shape.Length - 2; i >= 0; i--)
            {
                originalStrides[i] = originalStrides[i + 1] * shape[i + 1];
            }
        }

        for (var i = 0; i < rank; i++)
        {
            if (i < offset)
            {
                strides[i] = 0; // This dimension doesn't exist in original - broadcast
            }
            else
            {
                var origDim = i - offset;
                strides[i] = shape[origDim] == 1 ? 0 : originalStrides[origDim];
            }
        }

        return strides;
    }
}
