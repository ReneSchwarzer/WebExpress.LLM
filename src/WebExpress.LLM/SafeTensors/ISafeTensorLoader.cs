using System;
using System.Collections.Generic;

namespace WebExpress.LLM.SafeTensors;

/// <summary>
/// Provides a common interface for loading tensors from SafeTensors files,
/// supporting both single-file and sharded (multi-file) weight storage.
/// </summary>
public interface ISafeTensorLoader
{
    /// <summary>
    /// Gets the names of all tensors available through this loader.
    /// </summary>
    IReadOnlyCollection<string> TensorNames { get; }

    /// <summary>
    /// Gets the metadata for the specified tensor.
    /// </summary>
    /// <param name="name">The name of the tensor.</param>
    /// <returns>The tensor metadata.</returns>
    /// <exception cref="KeyNotFoundException">Thrown when the tensor name is not found.</exception>
    TensorMetadata GetMetadata(string name);

    /// <summary>
    /// Checks whether a tensor with the given name exists.
    /// </summary>
    /// <param name="name">The name of the tensor.</param>
    /// <returns>True if the tensor exists; otherwise, false.</returns>
    bool ContainsTensor(string name);

    /// <summary>
    /// Loads a tensor as a float array, converting from the stored data type if necessary.
    /// </summary>
    /// <param name="name">The name of the tensor to load.</param>
    /// <returns>A <see cref="Tensor.Tensor"/> containing the tensor data as float32.</returns>
    /// <exception cref="KeyNotFoundException">Thrown when the tensor name is not found.</exception>
    Tensor.Tensor LoadTensor(string name);

    /// <summary>
    /// Loads a single row from a 2-D tensor directly into the destination buffer.
    /// Intended for very large tensors where loading the full tensor at once is not feasible.
    /// </summary>
    /// <param name="name">The tensor name.</param>
    /// <param name="rowIndex">Zero-based row index. Uses <see cref="long"/> for very large tensors.</param>
    /// <param name="destination">Destination buffer that receives one full row.</param>
    void LoadTensorRow(string name, long rowIndex, float[] destination)
    {
        ArgumentNullException.ThrowIfNull(destination);

        var tensor = LoadTensor(name);

        if (tensor.Rank != 2)
        {
            throw new InvalidOperationException($"Tensor '{name}' is not 2-D.");
        }

        if (rowIndex < 0 || rowIndex >= tensor.Shape[0])
        {
            throw new ArgumentOutOfRangeException(nameof(rowIndex));
        }

        var cols = tensor.Shape[1];

        if (destination.Length < cols)
        {
            throw new ArgumentException(
                $"Destination length {destination.Length} is smaller than row width {cols}.",
                nameof(destination));
        }

        Array.Copy(tensor.Data, (int)rowIndex * cols, destination, 0, cols);
    }

    /// <summary>
    /// Loads the tensor with the specified name if it exists, or returns <c>null</c> otherwise.
    /// </summary>
    /// <param name="name">The name of the tensor to load.</param>
    /// <returns>The loaded tensor, or <c>null</c> if no tensor with the given name exists.</returns>
    Tensor.Tensor TryLoadTensor(string name)
    {
        return ContainsTensor(name) ? LoadTensor(name) : null;
    }
}
