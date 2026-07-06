using WebExpress.LLM.Gemma;
using WebExpress.LLM.Tensor;

namespace WebExpress.LLM.Test.Gemma;

/// <summary>
/// Provides unit tests for the KvCache class, ensuring correct storage and retrieval of key-value pairs.
/// </summary>
public sealed class UnitTestKvCache
{
    /// <summary>
    /// Tests that the constructor creates an empty cache.
    /// </summary>
    [Fact]
    public void Constructor_ShouldCreateEmptyCache()
    {
        var cache = new KvCache();

        Assert.Equal(0, cache.LayerCount);
        Assert.False(cache.HasLayer(0));
    }

    /// <summary>
    /// Tests that the update method stores keys and values.
    /// </summary>
    [Fact]
    public void Update_ShouldStoreKeysAndValues()
    {
        var cache = new KvCache();

        var keys = new WebExpress.LLM.Tensor.Tensor([2, 3, 4], new float[24]);
        var values = new WebExpress.LLM.Tensor.Tensor([2, 3, 4], new float[24]);

        cache.Update(0, keys, values);

        Assert.True(cache.HasLayer(0));
        Assert.Equal(1, cache.LayerCount);
        Assert.Equal(3, cache.GetSequenceLength(0));
    }

    /// <summary>
    /// Tests that the update method appends to an existing cache.
    /// </summary>
    [Fact]
    public void Update_ShouldAppendToExistingCache()
    {
        var cache = new KvCache();

        // First update: 2 heads, 3 positions, 4 dims
        var keys1 = new WebExpress.LLM.Tensor.Tensor([2, 3, 4], new float[24]);
        var values1 = new WebExpress.LLM.Tensor.Tensor([2, 3, 4], new float[24]);
        cache.Update(0, keys1, values1);

        Assert.Equal(3, cache.GetSequenceLength(0));

        // Second update: 2 heads, 1 position, 4 dims
        var keys2 = new WebExpress.LLM.Tensor.Tensor([2, 1, 4], new float[8]);
        var values2 = new WebExpress.LLM.Tensor.Tensor([2, 1, 4], new float[8]);
        cache.Update(0, keys2, values2);

        Assert.Equal(4, cache.GetSequenceLength(0));
    }

    /// <summary>
    /// Tests that the get method returns cached data.
    /// </summary>
    [Fact]
    public void Get_ShouldReturnCachedData()
    {
        var cache = new KvCache();

        var keysData = new float[8]; // [2, 1, 4]
        keysData[0] = 1.0f; // First element
        var keys = new WebExpress.LLM.Tensor.Tensor([2, 1, 4], keysData);

        var valuesData = new float[8];
        valuesData[0] = 2.0f;
        var values = new WebExpress.LLM.Tensor.Tensor([2, 1, 4], valuesData);

        cache.Update(5, keys, values);

        var (cachedKeys, cachedValues) = cache.Get(5);

        Assert.Equal(2, cachedKeys.Shape[0]);
        Assert.Equal(1, cachedKeys.Shape[1]);
        Assert.Equal(4, cachedKeys.Shape[2]);
        Assert.Equal(1.0f, cachedKeys[0, 0, 0]);
        Assert.Equal(2.0f, cachedValues[0, 0, 0]);
    }

    /// <summary>
    /// Tests that the get method throws an exception for a non-existent layer.
    /// </summary>
    [Fact]
    public void Get_NonexistentLayer_ShouldThrow()
    {
        var cache = new KvCache();
        Assert.Throws<KeyNotFoundException>(() => cache.Get(0));
    }

    /// <summary>
    /// Tests that getting the sequence length of an empty layer returns zero.
    /// </summary>
    [Fact]
    public void GetSequenceLength_EmptyLayer_ShouldReturnZero()
    {
        var cache = new KvCache();
        Assert.Equal(0, cache.GetSequenceLength(99));
    }

    /// <summary>
    /// Tests that the clear method removes all data from the cache.
    /// </summary>
    [Fact]
    public void Clear_ShouldRemoveAllData()
    {
        var cache = new KvCache();

        cache.Update(0, new WebExpress.LLM.Tensor.Tensor([1, 2, 4], new float[8]),
                        new WebExpress.LLM.Tensor.Tensor([1, 2, 4], new float[8]));
        cache.Update(1, new WebExpress.LLM.Tensor.Tensor([1, 2, 4], new float[8]),
                        new WebExpress.LLM.Tensor.Tensor([1, 2, 4], new float[8]));

        Assert.Equal(2, cache.LayerCount);

        cache.Clear();

        Assert.Equal(0, cache.LayerCount);
        Assert.False(cache.HasLayer(0));
        Assert.False(cache.HasLayer(1));
    }

    /// <summary>
    /// Tests that updating multiple layers tracks them independently.
    /// </summary>
    [Fact]
    public void Update_MultipleLayers_ShouldTrackIndependently()
    {
        var cache = new KvCache();

        cache.Update(0, new WebExpress.LLM.Tensor.Tensor([1, 3, 4], new float[12]),
                        new WebExpress.LLM.Tensor.Tensor([1, 3, 4], new float[12]));
        cache.Update(1, new WebExpress.LLM.Tensor.Tensor([1, 5, 4], new float[20]),
                        new WebExpress.LLM.Tensor.Tensor([1, 5, 4], new float[20]));

        Assert.Equal(3, cache.GetSequenceLength(0));
        Assert.Equal(5, cache.GetSequenceLength(1));
    }

    /// <summary>
    /// Tests that the update method clones the data.
    /// </summary>
    [Fact]
    public void Update_ShouldCloneData()
    {
        var cache = new KvCache();

        var keysData = new float[] { 1, 2, 3, 4, 5, 6, 7, 8 };
        var keys = new WebExpress.LLM.Tensor.Tensor([2, 1, 4], keysData);
        var values = new WebExpress.LLM.Tensor.Tensor([2, 1, 4], new float[8]);

        cache.Update(0, keys, values);

        // Modify original - cache should be unaffected
        keys[0] = 999;

        var (cachedKeys, _) = cache.Get(0);
        Assert.Equal(1.0f, cachedKeys[0, 0, 0]);
    }

    /// <summary>
    /// Verifies that the cache does not allocate per-token during normal generation:
    /// after appending 1024 tokens, the backing buffer length equals the reserved
    /// capacity (no per-token growth).
    /// </summary>
    [Fact]
    public void Append_WithinReservedCapacity_ShouldNotGrowBuffer()
    {
        var cache = new KvCache();
        cache.Reserve(layerIndex: 0, numHeads: 4, headDim: 8, capacity: 1024);

        // Append 1024 single-position updates.
        for (var i = 0; i < 1024; i++)
        {
            var keys = new float[4 * 8];
            var values = new float[4 * 8];
            cache.Append(0, keys, values);
        }

        Assert.Equal(1024, cache.GetSequenceLength(0));

        var (cachedKeys, cachedValues) = cache.Get(0);
        // After 1024 appends the backing buffer length must still equal the reserved
        // capacity (1024 positions * 4 heads * 8 head_dim = 32768 floats) — proves
        // no per-token reallocation happened.
        var backingFloats = 1024 * 4 * 8;
        Assert.Equal(backingFloats, cachedKeys.Data.Length);
        Assert.Equal(backingFloats, cachedValues.Data.Length);
    }

    /// <summary>
    /// Verifies that the buffer doubles on demand when capacity is exceeded and the
    /// live region is preserved across the grow.
    /// </summary>
    [Fact]
    public void Append_BeyondReservedCapacity_ShouldGrowAndPreserveLiveData()
    {
        var cache = new KvCache();
        cache.Reserve(layerIndex: 0, numHeads: 2, headDim: 4, capacity: 4);

        // Cache stores K/V with shape [numHeads, seqLen, headDim] flattened in
        // head-major order (matches the layout produced by MultiHeadAttention.ReshapeToHeads
        // and consumed by ComputeAttentionScores). For a single-position append with
        // numHeads=2, headDim=4 the flat index for (head=0, seq=0, dim=0) is 0; for
        // (head=1, seq=0, dim=0) it is 4.
        for (var i = 0; i < 4; i++)
        {
            var keys = new float[2 * 4];
            var values = new float[2 * 4];
            // Encode position index i into both head-0-dim-0 and head-1-dim-0 so we can
            // assert with the [head, seq, dim] indexer after the grow.
            keys[0] = i;          // head=0, seq=0 (within this slice), dim=0
            keys[4] = i + 50;     // head=1, seq=0, dim=0
            values[0] = i + 100;
            values[4] = i + 200;
            cache.Append(0, keys, values);
        }

        Assert.Equal(4, cache.GetSequenceLength(0));

        // One more append must grow the buffer.
        var extraKeys = new float[2 * 4];
        var extraValues = new float[2 * 4];
        extraKeys[0] = 99;
        extraKeys[4] = 149;
        extraValues[0] = 199;
        extraValues[4] = 249;
        cache.Append(0, extraKeys, extraValues);

        Assert.Equal(5, cache.GetSequenceLength(0));

        var (cachedKeys, cachedValues) = cache.Get(0);
        // Live region preserved through the grow. Indexer is [head, seq, dim]; flat stride
        // for seq is numHeads*headDim=8, for head is headDim=4.
        Assert.Equal(0f, cachedKeys[0, 0, 0]);
        Assert.Equal(1f, cachedKeys[0, 1, 0]);
        Assert.Equal(2f, cachedKeys[0, 2, 0]);
        Assert.Equal(3f, cachedKeys[0, 3, 0]);
        Assert.Equal(99f, cachedKeys[0, 4, 0]);
        Assert.Equal(50f, cachedKeys[1, 0, 0]);
        Assert.Equal(51f, cachedKeys[1, 1, 0]);
        Assert.Equal(52f, cachedKeys[1, 2, 0]);
        Assert.Equal(53f, cachedKeys[1, 3, 0]);
        Assert.Equal(100f, cachedValues[0, 0, 0]);
        Assert.Equal(199f, cachedValues[0, 4, 0]);
        Assert.Equal(249f, cachedValues[1, 4, 0]);
    }

    /// <summary>
    /// Verifies that Clear() releases the buffers and the cache accepts a fresh
    /// reservation with a different shape afterwards.
    /// </summary>
    [Fact]
    public void Clear_AfterReserve_AllowsFreshReservation()
    {
        var cache = new KvCache();
        cache.Reserve(0, 2, 4, 16);
        cache.Append(0, new float[2 * 4], new float[2 * 4]);

        cache.Clear();

        Assert.Equal(0, cache.LayerCount);
        Assert.Equal(0, cache.GetSequenceLength(0));

        // After Clear, the cache must accept a fresh reservation with different shape.
        cache.Reserve(0, 3, 5, 8);
        Assert.Equal(0, cache.GetSequenceLength(0));
        Assert.True(cache.HasLayer(0));

        // And the new shape must work for appends.
        cache.Append(0, new float[3 * 5], new float[3 * 5]);
        Assert.Equal(1, cache.GetSequenceLength(0));
    }
}
