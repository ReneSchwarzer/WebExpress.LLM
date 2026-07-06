using System;
using System.Collections.Generic;
using WebExpress.LLM.Tensor;

namespace WebExpress.LLM.Gemma;

/// <summary>
/// Implements the key-value cache used during autoregressive generation to avoid
/// recomputing attention over previously processed tokens.
/// </summary>
/// <remarks>
/// Each transformer layer maintains its own pre-allocated key and value buffers. During
/// generation, the new token's keys/values are appended in place at the end of the buffer,
/// while the full cached keys/values are exposed as zero-copy views. Pre-allocating a
/// single contiguous buffer per layer avoids the O(n²) memory traffic that repeated
/// <see cref="TensorOperations.Concatenate"/> calls would incur over a long generation.
///
/// The buffer for layer <c>i</c> is allocated lazily on the first
/// <see cref="Append(int, ReadOnlySpan{float}, ReadOnlySpan{float})"/> call. The caller
/// supplies the capacity (number of positions) up front — typically the model's
/// <c>context_length</c> — so the buffer never re-allocates during normal generation.
/// If appends exceed the requested capacity the cache grows by doubling, matching the
/// standard amortised-O(1) growth strategy.
/// </remarks>
public sealed class KvCache
{
    /// <summary>Per-layer storage: backing buffer plus its logical length.</summary>
    private sealed class LayerBuffer
    {
        public float[] Keys;
        public float[] Values;
        public int NumHeads;
        public int HeadDim;
        public int Len;
        public int Capacity;
    }

    private readonly Dictionary<int, LayerBuffer> _cache = [];

    /// <summary>
    /// Initializes a new empty KV cache.
    /// </summary>
    public KvCache()
    {
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
        return _cache.TryGetValue(layerIndex, out var entry) ? entry.Len : 0;
    }

    /// <summary>
    /// Reserves storage for <paramref name="capacity"/> positions on <paramref name="layerIndex"/>.
    /// Subsequent <see cref="Append(int, ReadOnlySpan{float}, ReadOnlySpan{float})"/> calls
    /// will not re-allocate until the reserved capacity is exceeded.
    /// </summary>
    /// <param name="layerIndex">The transformer layer index.</param>
    /// <param name="numHeads">Number of KV heads per position (e.g. <c>num_key_value_heads</c>).</param>
    /// <param name="headDim">Per-head dimension (e.g. <c>head_dim</c>).</param>
    /// <param name="capacity">Maximum number of positions the buffer will hold before growing.</param>
    public void Reserve(int layerIndex, int numHeads, int headDim, int capacity)
    {
        if (numHeads <= 0) throw new ArgumentOutOfRangeException(nameof(numHeads), "numHeads must be positive.");
        if (headDim <= 0) throw new ArgumentOutOfRangeException(nameof(headDim), "headDim must be positive.");
        if (capacity <= 0) throw new ArgumentOutOfRangeException(nameof(capacity), "capacity must be positive.");

        if (_cache.TryGetValue(layerIndex, out var existing))
        {
            // Capacity is sticky once allocated: only raise it.
            if (capacity > existing.Capacity)
            {
                existing.Keys = GrowBufferHeadOuter(
                    existing.Keys, existing.Capacity, capacity,
                    numHeads, headDim, existing.Len);
                existing.Values = GrowBufferHeadOuter(
                    existing.Values, existing.Capacity, capacity,
                    numHeads, headDim, existing.Len);
                existing.Capacity = capacity;
            }

            // numHeads/headDim changes are not supported (would change the buffer layout).
            if (existing.NumHeads != numHeads || existing.HeadDim != headDim)
            {
                throw new InvalidOperationException(
                    $"KV cache layer {layerIndex} was reserved with shape ({existing.NumHeads},{existing.HeadDim}) " +
                    $"but a new reservation requested ({numHeads},{headDim}). Shape changes are not supported.");
            }

            return;
        }

        var perPos = numHeads * headDim;
        _cache[layerIndex] = new LayerBuffer
        {
            Keys = new float[capacity * perPos],
            Values = new float[capacity * perPos],
            NumHeads = numHeads,
            HeadDim = headDim,
            Len = 0,
            Capacity = capacity,
        };
    }

    /// <summary>
    /// Appends new keys and values to the cached buffer for the given layer.
    /// </summary>
    /// <remarks>
    /// The spans must be laid out as <c>[numHeads, newLen, headDim]</c> with the last
    /// dimension contiguous (matching the layout produced by <c>MultiHeadAttention</c>'s
    /// head reshape and by <see cref="TensorOperations.Concatenate"/> along dim=1).
    /// Reserves the buffer on first call if <see cref="Reserve(int, int, int, int)"/>
    /// has not yet been invoked, deriving capacity from <paramref name="newKeys"/>.
    /// </remarks>
    /// <param name="layerIndex">The transformer layer index.</param>
    /// <param name="newKeys">New keys with shape <c>[numHeads, newLen, headDim]</c>.</param>
    /// <param name="newValues">New values with shape <c>[numHeads, newLen, headDim]</c>.</param>
    public void Append(int layerIndex, ReadOnlySpan<float> newKeys, ReadOnlySpan<float> newValues)
    {
        if (!_cache.TryGetValue(layerIndex, out var entry))
        {
            // No reservation yet: derive shape from the incoming data. The total element
            // count must equal numHeads * headDim * newLen, but we don't know newLen here.
            // Fall back to allocating capacity for exactly newLen positions; this branch is
            // only used when callers skip Reserve(). Pathological; emit a clear error.
            throw new InvalidOperationException(
                $"KV cache layer {layerIndex} has no reserved storage. " +
                $"Call Reserve(layerIndex, numHeads, headDim, capacity) before Append().");
        }

        if (newKeys.Length != newValues.Length)
        {
            throw new ArgumentException(
                $"Key/value span lengths differ ({newKeys.Length} vs {newValues.Length}).");
        }

        var perPos = entry.NumHeads * entry.HeadDim;

        if (newKeys.Length == 0 || newKeys.Length % perPos != 0)
        {
            throw new ArgumentException(
                $"Incoming key span length {newKeys.Length} is not a multiple of numHeads*headDim ({perPos}).");
        }

        var newLen = newKeys.Length / perPos;
        var requiredCapacity = entry.Len + newLen;

        if (requiredCapacity > entry.Capacity)
        {
            // Grow by doubling, copying the live region to the front of the new buffer.
            // The live region is copied in its head-outer layout.
            var newCapacity = Math.Max(requiredCapacity, entry.Capacity * 2);
            entry.Keys = GrowBufferHeadOuter(entry.Keys, entry.Capacity, newCapacity, entry.NumHeads, entry.HeadDim, entry.Len);
            entry.Values = GrowBufferHeadOuter(entry.Values, entry.Capacity, newCapacity, entry.NumHeads, entry.HeadDim, entry.Len);
            entry.Capacity = newCapacity;
        }

        // Append in head-outer order. The incoming span is laid out [numHeads, newLen, headDim]
        // flattened head-outer (same convention as MultiHeadAttention.ReshapeToHeads and
        // TensorOperations.Concatenate along dim=1); we copy directly into the matching
        // head-outer storage slots. When newLen == 1 (incremental decoding) this is a
        // contiguous copy per head; when newLen > 1 (prefill) it is still contiguous.
        var headDim = entry.HeadDim;
        for (var h = 0; h < entry.NumHeads; h++)
        {
            var srcHeadOffset = h * newLen * headDim;
            var dstHeadOffset = h * entry.Capacity * headDim + entry.Len * headDim;
            newKeys.Slice(srcHeadOffset, newLen * headDim)
                .CopyTo(entry.Keys.AsSpan(dstHeadOffset, newLen * headDim));
            newValues.Slice(srcHeadOffset, newLen * headDim)
                .CopyTo(entry.Values.AsSpan(dstHeadOffset, newLen * headDim));
        }
        entry.Len += newLen;
    }

    /// <summary>
    /// Allocates a new buffer of size <paramref name="newCapacity"/> positions in head-outer
    /// layout and copies the first <paramref name="liveLen"/> positions from
    /// <paramref name="source"/>, which itself is head-outer with stride
    /// <paramref name="oldCapacity"/> * <paramref name="headDim"/>.
    /// </summary>
    private static float[] GrowBufferHeadOuter(float[] source, int oldCapacity, int newCapacity, int numHeads, int headDim, int liveLen)
    {
        var grown = new float[newCapacity * numHeads * headDim];
        if (liveLen > 0)
        {
            for (var h = 0; h < numHeads; h++)
            {
                Array.Copy(source, h * oldCapacity * headDim, grown,
                    h * newCapacity * headDim, liveLen * headDim);
            }
        }
        return grown;
    }

    /// <summary>
    /// Gets the cached keys and values for the specified layer as views into the
    /// pre-allocated buffer (zero-copy).
    /// </summary>
    /// <remarks>
    /// The returned tensors share storage with the cache; their lifetime is bounded by
    /// the cache's lifetime. Mutations made through the returned tensors (e.g. from a
    /// subsequent <see cref="Tensor.Clone"/>) are detached from the cache. The tensors
    /// themselves are read-only references — the caller must not assume the cache will
    /// not re-grow between the call and tensor use.
    /// </remarks>
    /// <param name="layerIndex">The transformer layer index.</param>
    /// <returns>A tuple of (Keys, Values) tensors of shape <c>[numHeads, len, headDim]</c>.</returns>
    /// <exception cref="KeyNotFoundException">Thrown when no cache exists for the specified layer.</exception>
    public (Tensor.Tensor Keys, Tensor.Tensor Values) Get(int layerIndex)
    {
        if (!_cache.TryGetValue(layerIndex, out var entry))
        {
            throw new KeyNotFoundException($"No cache entry for layer {layerIndex}.");
        }

        // Storage is head-outer: each head owns a contiguous region of
        // capacity*headDim floats in the backing buffer. The view exposed to callers
        // uses shape [numHeads, len, headDim] where len is the live sequence length
        // (so the view strides match the storage strides exactly when stride for heads
        // is capacity*headDim and stride for seq is headDim). Offset is 0.
        var shape = new[] { entry.NumHeads, entry.Len, entry.HeadDim };
        var strides = new[] { entry.Capacity * entry.HeadDim, entry.HeadDim, 1 };
        var keys = new Tensor.Tensor(shape, entry.Keys, 0, entry.NumHeads * entry.Len * entry.HeadDim, strides);
        var values = new Tensor.Tensor(shape, entry.Values, 0, entry.NumHeads * entry.Len * entry.HeadDim, strides);
        return (keys, values);
    }

    /// <summary>
    /// Backwards-compatible tensor-based append. Allocates a fresh contiguous buffer
    /// copy of <paramref name="newKeys"/>/<paramref name="newValues"/> and forwards to
    /// the span-based <see cref="Append(int, ReadOnlySpan{float}, ReadOnlySpan{float})"/>.
    /// Prefer the span overload in hot paths.
    /// </summary>
    /// <param name="layerIndex">The transformer layer index.</param>
    /// <param name="newKeys">New keys with shape <c>[numHeads, newLen, headDim]</c>.</param>
    /// <param name="newValues">New values with shape <c>[numHeads, newLen, headDim]</c>.</param>
    public void Update(int layerIndex, Tensor.Tensor newKeys, Tensor.Tensor newValues)
    {
        ArgumentNullException.ThrowIfNull(newKeys);
        ArgumentNullException.ThrowIfNull(newValues);

        // Lazy reservation from the incoming tensor shape: pull numHeads/headDim from
        // the tensor and use newLen as the capacity. This keeps Update() self-sufficient
        // for callers that don't know the model context length up front (e.g. tests).
        if (!_cache.ContainsKey(layerIndex))
        {
            if (newKeys.Rank != 3)
            {
                throw new ArgumentException(
                    $"KV cache Update requires 3D tensors [numHeads, newLen, headDim], got rank {newKeys.Rank}.");
            }

            var numHeads = newKeys.Shape[0];
            var newLen = newKeys.Shape[1];
            var headDim = newKeys.Shape[2];
            Reserve(layerIndex, numHeads, headDim, newLen);
        }

        Append(layerIndex, newKeys.DataSpan, newValues.DataSpan);
    }

    /// <summary>
    /// Checks whether the cache contains data for the specified layer.
    /// </summary>
    public bool HasLayer(int layerIndex)
    {
        return _cache.ContainsKey(layerIndex);
    }

    /// <summary>
    /// Clears all cached data for all layers. Buffers are released for GC; the next
    /// <see cref="Reserve(int, int, int, int)"/> or <see cref="Append(int, ReadOnlySpan{float}, ReadOnlySpan{float})"/>
    /// call will allocate fresh storage.
    /// </summary>
    public void Clear()
    {
        _cache.Clear();
    }

}