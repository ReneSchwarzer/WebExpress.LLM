using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using WebExpress.LLM.Model;

namespace WebExpress.LLM.SafeTensors;

/// <summary>
/// Loads tensors from sharded SafeTensors files, where model weights are distributed
/// across multiple shard files as described by a <see cref="SafeTensorIndex"/>.
/// </summary>
/// <remarks>
/// Each shard file is a standard SafeTensors binary file. The index maps tensor names
/// to their respective shard files. This loader creates one <see cref="SafeTensorLoader"/>
/// per shard and delegates tensor operations to the appropriate shard loader.
/// </remarks>
public sealed class ShardedSafeTensorLoader : ISafeTensorLoader, IDisposable
{
    private readonly SafeTensorIndex _index;
    private readonly Dictionary<string, SafeTensorLoader> _shardLoaders;
    private readonly Dictionary<string, ModelWeights> _shardWeights;
    private readonly IReadOnlyCollection<string> _tensorNames;
    private bool _disposed;

    /// <summary>
    /// Initializes a new ShardedSafeTensorLoader from an index and a directory containing shard files.
    /// </summary>
    /// <param name="index">The parsed SafeTensors index describing the tensor-to-shard mapping.</param>
    /// <param name="modelDirectory">The directory containing the shard files.</param>
    /// <exception cref="ArgumentNullException">Thrown when index is null.</exception>
    /// <exception cref="ArgumentException">Thrown when modelDirectory is null or whitespace.</exception>
    /// <exception cref="FileNotFoundException">Thrown when a referenced shard file does not exist.</exception>
    public ShardedSafeTensorLoader(SafeTensorIndex index, string modelDirectory)
    {
        _index = index ?? throw new ArgumentNullException(nameof(index));

        if (string.IsNullOrWhiteSpace(modelDirectory))
        {
            throw new ArgumentException("Model directory must be provided.", nameof(modelDirectory));
        }

        _shardLoaders = new Dictionary<string, SafeTensorLoader>(StringComparer.Ordinal);
        _shardWeights = new Dictionary<string, ModelWeights>(StringComparer.Ordinal);

        foreach (var shardFile in index.ShardFiles)
        {
            var shardPath = Path.Combine(modelDirectory, shardFile);

            if (!File.Exists(shardPath))
            {
                throw new FileNotFoundException(
                    $"Shard file '{shardFile}' referenced by the index was not found.", shardPath);
            }

            var weights = ModelWeights.FromFile(shardPath);
            _shardWeights[shardFile] = weights;
            _shardLoaders[shardFile] = new SafeTensorLoader(weights);
        }

        RecomputeShardBaseOffsets();

        _tensorNames = index.WeightMap.Keys.ToList().AsReadOnly();
    }

    /// <summary>
    /// Initializes a new ShardedSafeTensorLoader from an index and pre-loaded shard loaders.
    /// This constructor is primarily intended for testing purposes.
    /// </summary>
    /// <param name="index">The parsed SafeTensors index describing the tensor-to-shard mapping.</param>
    /// <param name="shardLoaders">A dictionary mapping shard filenames to their SafeTensorLoader instances.</param>
    /// <exception cref="ArgumentNullException">Thrown when index or shardLoaders is null.</exception>
    /// <exception cref="InvalidDataException">Thrown when a shard referenced in the index is missing from the loaders.</exception>
    public ShardedSafeTensorLoader(SafeTensorIndex index, Dictionary<string, SafeTensorLoader> shardLoaders)
    {
        _index = index ?? throw new ArgumentNullException(nameof(index));

        if (shardLoaders == null)
        {
            throw new ArgumentNullException(nameof(shardLoaders));
        }

        _shardLoaders = new Dictionary<string, SafeTensorLoader>(shardLoaders, StringComparer.Ordinal);
        _shardWeights = new Dictionary<string, ModelWeights>(StringComparer.Ordinal);

        foreach (var shardFile in index.ShardFiles)
        {
            if (!_shardLoaders.ContainsKey(shardFile))
            {
                throw new InvalidDataException(
                    $"Shard file '{shardFile}' referenced by the index is missing from the provided loaders.");
            }
        }

        RecomputeShardBaseOffsets();

        _tensorNames = index.WeightMap.Keys.ToList().AsReadOnly();
    }

    /// <summary>
    /// Gets the names of all tensors across all shards.
    /// </summary>
    public IReadOnlyCollection<string> TensorNames => _tensorNames;

    /// <summary>
    /// Gets the parsed index containing metadata and the weight map.
    /// </summary>
    public SafeTensorIndex Index => _index;

    /// <summary>
    /// Gets the metadata for the specified tensor from the appropriate shard.
    /// </summary>
    /// <param name="name">The name of the tensor.</param>
    /// <returns>The tensor metadata.</returns>
    /// <exception cref="KeyNotFoundException">Thrown when the tensor name is not found.</exception>
    public TensorMetadata GetMetadata(string name)
    {
        var loader = GetShardLoader(name);

        return loader.GetMetadata(name);
    }

    /// <summary>
    /// Checks whether a tensor with the given name exists in any shard.
    /// </summary>
    /// <param name="name">The name of the tensor.</param>
    /// <returns>True if the tensor exists; otherwise, false.</returns>
    public bool ContainsTensor(string name)
    {
        return _index.WeightMap.ContainsKey(name);
    }

    /// <summary>
    /// Loads a tensor from the appropriate shard, converting to float32 if necessary.
    /// </summary>
    /// <param name="name">The name of the tensor to load.</param>
    /// <returns>A <see cref="Tensor.Tensor"/> containing the tensor data as float32.</returns>
    /// <exception cref="KeyNotFoundException">Thrown when the tensor name is not found.</exception>
    public Tensor.Tensor LoadTensor(string name)
    {
        var loader = GetShardLoader(name);

        return loader.LoadTensor(name);
    }

    /// <summary>
    /// Releases all resources held by the shard loaders and their underlying weights.
    /// </summary>
    public void Dispose()
    {
        if (_disposed)
        {
            return;
        }

        foreach (var weights in _shardWeights.Values)
        {
            weights.Dispose();
        }

        _shardWeights.Clear();
        _shardLoaders.Clear();
        _disposed = true;
    }

    /// <summary>
    /// Retrieves the loader associated with the specified tensor name from the SafeTensors index.
    /// </summary>
    /// <param name="tensorName">
    /// The name of the tensor for which to retrieve the corresponding shard loader. Cannot be null or empty.
    /// </param>
    /// <returns>
    /// The loader instance responsible for loading the specified tensor's shard file.
    /// </returns>
    /// <exception cref="KeyNotFoundException">
    /// Thrown if the specified tensor name does not exist in the SafeTensors index.
    /// </exception>
    /// <exception cref="InvalidDataException">
    /// Thrown if the shard file associated with the tensor name is not loaded.
    /// </exception>
    private SafeTensorLoader GetShardLoader(string tensorName)
    {
        if (!_index.WeightMap.TryGetValue(tensorName, out var shardFile))
        {
            throw new KeyNotFoundException(
                $"Tensor '{tensorName}' not found in the SafeTensors index.");
        }

        if (!_shardLoaders.TryGetValue(shardFile, out var loader))
        {
            throw new InvalidDataException(
                $"Shard file '{shardFile}' for tensor '{tensorName}' is not loaded.");
        }

        return loader;
    }

    /// <summary>
    /// Recomputes the base offset for each shard using only the tensors that
    /// the index maps to that shard. This avoids the problem where a shard's
    /// header lists tensors from other shards whose offsets would poison the
    /// minimum base offset calculation in <see cref="SafeTensorLoader"/>.
    /// </summary>
    private void RecomputeShardBaseOffsets()
    {
        // Group tensor names by shard file
        var tensorsByShardFile = _index.WeightMap
            .GroupBy(kv => kv.Value, kv => kv.Key, StringComparer.Ordinal);

        foreach (var group in tensorsByShardFile)
        {
            var shardFile = group.Key;

            if (!_shardLoaders.TryGetValue(shardFile, out var loader))
            {
                continue;
            }

            var dataSize = loader.DataSectionSize;
            long maxEnd = 0;
            var minBegin = long.MaxValue;

            foreach (var tensorName in group)
            {
                if (!loader.ContainsTensor(tensorName))
                {
                    continue;
                }

                var meta = loader.GetMetadata(tensorName);

                if (meta.DataOffsets.Count < 2)
                {
                    continue;
                }

                var begin = meta.DataOffsets[0];
                var end = meta.DataOffsets[1];

                // Skip zero-size tensors
                if (end <= begin)
                {
                    continue;
                }

                if (begin < minBegin)
                {
                    minBegin = begin;
                }

                if (end > maxEnd)
                {
                    maxEnd = end;
                }
            }

            // If all offsets fit within the data section, they're already local — no adjustment needed.
            if (maxEnd <= dataSize || minBegin == long.MaxValue)
            {
                loader.SetBaseOffset(0);
            }
            else
            {
                loader.SetBaseOffset(minBegin);
            }
        }
    }
}
