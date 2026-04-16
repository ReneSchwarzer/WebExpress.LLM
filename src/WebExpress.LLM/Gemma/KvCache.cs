using System;
using System.Collections.Generic;
using WebExpress.LLM.Tensor;

namespace WebExpress.LLM.Gemma;

/// <summary>
/// Implements the key-value cache used during autoregressive generation to avoid
/// recomputing attention over previously processed tokens.
/// </summary>
/// <remarks>
/// Each transformer layer maintains its own cached key and value tensors. During
/// generation, only the new token's keys/values are computed and appended to the cache,
/// while the full cached keys/values are used for attention computation.
/// </remarks>
public sealed class KvCache
{
    private readonly Dictionary<int, (Tensor.Tensor Keys, Tensor.Tensor Values)> _cache;

    /// <summary>
    /// Initializes a new empty KV cache.
    /// </summary>
    public KvCache()
    {
        _cache = new Dictionary<int, (Tensor.Tensor, Tensor.Tensor)>();
    }

    /// <summary>
    /// Gets the number of layers that currently have cached data.
    /// </summary>
    public int LayerCount => _cache.Count;

    /// <summary>
    /// Gets the current sequence length for the specified layer, or 0 if no cache exists.
    /// </summary>
    /// <param name="layerIndex">The transformer layer index.</param>
    /// <returns>The number of cached positions.</returns>
    public int GetSequenceLength(int layerIndex)
    {
        if (!_cache.TryGetValue(layerIndex, out var entry))
        {
            return 0;
        }

        // Keys shape: [numKvHeads, cachedLen, headDim]
        return entry.Keys.Shape[1];
    }

    /// <summary>
    /// Updates the cache for a given layer by appending new key and value tensors.
    /// </summary>
    /// <param name="layerIndex">The transformer layer index.</param>
    /// <param name="newKeys">New keys with shape [numKvHeads, newLen, headDim].</param>
    /// <param name="newValues">New values with shape [numKvHeads, newLen, headDim].</param>
    public void Update(int layerIndex, Tensor.Tensor newKeys, Tensor.Tensor newValues)
    {
        ArgumentNullException.ThrowIfNull(newKeys);
        ArgumentNullException.ThrowIfNull(newValues);

        if (_cache.TryGetValue(layerIndex, out var existing))
        {
            // Concatenate along the sequence dimension (dim=1)
            var concatenatedKeys = TensorOperations.Concatenate(existing.Keys, newKeys, dim: 1);
            var concatenatedValues = TensorOperations.Concatenate(existing.Values, newValues, dim: 1);
            _cache[layerIndex] = (concatenatedKeys, concatenatedValues);
        }
        else
        {
            _cache[layerIndex] = (newKeys.Clone(), newValues.Clone());
        }
    }

    /// <summary>
    /// Gets the cached keys and values for the specified layer.
    /// </summary>
    /// <param name="layerIndex">The transformer layer index.</param>
    /// <returns>A tuple of (Keys, Values) tensors.</returns>
    /// <exception cref="KeyNotFoundException">Thrown when no cache exists for the specified layer.</exception>
    public (Tensor.Tensor Keys, Tensor.Tensor Values) Get(int layerIndex)
    {
        if (!_cache.TryGetValue(layerIndex, out var entry))
        {
            throw new KeyNotFoundException($"No cache entry for layer {layerIndex}.");
        }

        return entry;
    }

    /// <summary>
    /// Checks whether the cache contains data for the specified layer.
    /// </summary>
    public bool HasLayer(int layerIndex)
    {
        return _cache.ContainsKey(layerIndex);
    }

    /// <summary>
    /// Clears all cached data for all layers.
    /// </summary>
    public void Clear()
    {
        _cache.Clear();
    }
}
