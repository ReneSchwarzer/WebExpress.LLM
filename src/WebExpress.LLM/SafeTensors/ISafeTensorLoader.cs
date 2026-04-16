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
}
