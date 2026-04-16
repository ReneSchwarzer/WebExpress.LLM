using WebExpress.LLM.Gemma;
using WebExpress.LLM.Tensor;

namespace WebExpress.LLM.Test.Gemma;

public sealed class KvCacheTests
{
    [Fact]
    public void Constructor_ShouldCreateEmptyCache()
    {
        var cache = new KvCache();

        Assert.Equal(0, cache.LayerCount);
        Assert.False(cache.HasLayer(0));
    }

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

    [Fact]
    public void Get_NonexistentLayer_ShouldThrow()
    {
        var cache = new KvCache();
        Assert.Throws<KeyNotFoundException>(() => cache.Get(0));
    }

    [Fact]
    public void GetSequenceLength_EmptyLayer_ShouldReturnZero()
    {
        var cache = new KvCache();
        Assert.Equal(0, cache.GetSequenceLength(99));
    }

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
}
