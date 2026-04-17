using System.Buffers.Binary;
using System.Text;
using System.Text.Json;
using WebExpress.LLM.Model;
using WebExpress.LLM.SafeTensors;

namespace WebExpress.LLM.Test.SafeTensors;

/// <summary>
/// Provides unit tests for the <see cref="SafeTensorLoader"/> class, covering tensor loading
/// correctness and the in-memory cache behavior.
/// </summary>
public sealed class UnitTestSafeTensorLoader
{
    #region LoadTensor correctness

    /// <summary>
    /// Tests that a single F32 tensor can be loaded and its values are correct.
    /// </summary>
    [Fact]
    public void LoadTensor_F32_ShouldReturnCorrectValues()
    {
        var data = new float[] { 1.0f, 2.0f, 3.0f, 4.0f };
        var file = BuildSafeTensorsFile([("weight", "F32", [2, 2], data)]);

        using var weights = ModelWeights.FromFile(file);
        var loader = new SafeTensorLoader(weights);

        var tensor = loader.LoadTensor("weight");

        Assert.Equal([2, 2], tensor.Shape);
        Assert.Equal(1.0f, tensor.Data[0], precision: 5);
        Assert.Equal(2.0f, tensor.Data[1], precision: 5);
        Assert.Equal(3.0f, tensor.Data[2], precision: 5);
        Assert.Equal(4.0f, tensor.Data[3], precision: 5);
    }

    /// <summary>
    /// Tests that a BF16 tensor is correctly converted to F32 on load.
    /// </summary>
    [Fact]
    public void LoadTensor_BF16_ShouldConvertToFloat32()
    {
        var values = new float[] { 1.0f, 2.0f };
        var file = BuildSafeTensorsFile([("w", "BF16", [1, 2], values)]);

        using var weights = ModelWeights.FromFile(file);
        var loader = new SafeTensorLoader(weights);

        var tensor = loader.LoadTensor("w");

        Assert.Equal([1, 2], tensor.Shape);
        Assert.Equal(1.0f, tensor.Data[0], precision: 2);
        Assert.Equal(2.0f, tensor.Data[1], precision: 2);
    }

    /// <summary>
    /// Tests that loading a tensor that does not exist throws a <see cref="KeyNotFoundException"/>.
    /// </summary>
    [Fact]
    public void LoadTensor_UnknownName_ShouldThrowKeyNotFoundException()
    {
        var file = BuildSafeTensorsFile([("weight", "F32", [1], [42.0f])]);

        using var weights = ModelWeights.FromFile(file);
        var loader = new SafeTensorLoader(weights);

        Assert.Throws<KeyNotFoundException>(() => loader.LoadTensor("nonexistent"));
    }

    #endregion

    #region Cache behavior

    /// <summary>
    /// Tests that calling LoadTensor twice for the same name returns the same object instance.
    /// </summary>
    [Fact]
    public void LoadTensor_SecondCall_ShouldReturnCachedInstance()
    {
        var file = BuildSafeTensorsFile([("weight", "F32", [4], [1.0f, 2.0f, 3.0f, 4.0f])]);

        using var weights = ModelWeights.FromFile(file);
        var loader = new SafeTensorLoader(weights);

        var first = loader.LoadTensor("weight");
        var second = loader.LoadTensor("weight");

        Assert.Same(first, second);
    }

    /// <summary>
    /// Tests that the cached tensor has the same data as the originally loaded one.
    /// </summary>
    [Fact]
    public void LoadTensor_CachedTensor_ShouldHaveSameData()
    {
        var data = new float[] { 5.0f, 10.0f, 15.0f };
        var file = BuildSafeTensorsFile([("w", "F32", [3], data)]);

        using var weights = ModelWeights.FromFile(file);
        var loader = new SafeTensorLoader(weights);

        var first = loader.LoadTensor("w");
        var second = loader.LoadTensor("w");

        Assert.Equal(first.Data, second.Data);
        Assert.Equal(first.Shape, second.Shape);
    }

    /// <summary>
    /// Tests that different tensor names are cached independently and return different instances.
    /// </summary>
    [Fact]
    public void LoadTensor_DifferentNames_ShouldCacheIndependently()
    {
        var file = BuildSafeTensorsFile(
        [
            ("alpha", "F32", [2], [1.0f, 2.0f]),
            ("beta",  "F32", [2], [3.0f, 4.0f])
        ]);

        using var weights = ModelWeights.FromFile(file);
        var loader = new SafeTensorLoader(weights);

        var alpha1 = loader.LoadTensor("alpha");
        var beta1  = loader.LoadTensor("beta");
        var alpha2 = loader.LoadTensor("alpha");
        var beta2  = loader.LoadTensor("beta");

        // Each name returns its own cached instance
        Assert.Same(alpha1, alpha2);
        Assert.Same(beta1, beta2);

        // The two names return different instances
        Assert.NotSame(alpha1, beta1);
    }

    /// <summary>
    /// Tests that the cached tensor contains the correct values after multiple calls.
    /// </summary>
    [Fact]
    public void LoadTensor_RepeatedCalls_ShouldReturnCorrectValuesEachTime()
    {
        var data = new float[] { 7.0f, 8.0f, 9.0f };
        var file = BuildSafeTensorsFile([("weights", "F32", [3], data)]);

        using var weights = ModelWeights.FromFile(file);
        var loader = new SafeTensorLoader(weights);

        for (var i = 0; i < 5; i++)
        {
            var tensor = loader.LoadTensor("weights");
            Assert.Equal(7.0f, tensor.Data[0], precision: 5);
            Assert.Equal(8.0f, tensor.Data[1], precision: 5);
            Assert.Equal(9.0f, tensor.Data[2], precision: 5);
        }
    }

    #endregion

    #region Helpers

    /// <summary>
    /// Builds a minimal SafeTensors binary file containing the specified tensors and returns its
    /// path in a temporary directory.
    /// </summary>
    private static string BuildSafeTensorsFile(
        IReadOnlyList<(string Name, string Dtype, long[] Shape, float[] Data)> tensors)
    {
        var tempDir = Path.Combine(Path.GetTempPath(), Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(tempDir);

        var filePath = Path.Combine(tempDir, "model.safetensors");

        // Build raw byte data for each tensor (F32 or BF16)
        var rawData = new Dictionary<string, byte[]>(StringComparer.Ordinal);
        foreach (var (name, dtype, _, data) in tensors)
        {
            rawData[name] = dtype == "BF16" ? FloatsToBF16Bytes(data) : FloatsToF32Bytes(data);
        }

        // Build JSON header
        var header = new Dictionary<string, object>(StringComparer.Ordinal);
        long offset = 0;

        foreach (var (name, dtype, shape, _) in tensors)
        {
            var byteLen = rawData[name].Length;
            header[name] = new
            {
                dtype,
                shape,
                data_offsets = new long[] { offset, offset + byteLen }
            };
            offset += byteLen;
        }

        var headerJson = JsonSerializer.Serialize(header);
        var headerBytes = Encoding.UTF8.GetBytes(headerJson);

        // Calculate total data size
        var totalData = rawData.Values.Sum(b => b.Length);
        var fileBytes = new byte[8 + headerBytes.Length + totalData];

        BinaryPrimitives.WriteInt64LittleEndian(fileBytes.AsSpan(0, 8), headerBytes.Length);
        headerBytes.CopyTo(fileBytes.AsSpan(8));

        var pos = 8 + headerBytes.Length;
        foreach (var (name, _, _, _) in tensors)
        {
            rawData[name].CopyTo(fileBytes.AsSpan(pos));
            pos += rawData[name].Length;
        }

        File.WriteAllBytes(filePath, fileBytes);

        return filePath;
    }

    private static byte[] FloatsToF32Bytes(float[] values)
    {
        var bytes = new byte[values.Length * 4];
        for (var i = 0; i < values.Length; i++)
        {
            BinaryPrimitives.WriteSingleLittleEndian(bytes.AsSpan(i * 4), values[i]);
        }
        return bytes;
    }

    private static byte[] FloatsToBF16Bytes(float[] values)
    {
        var bytes = new byte[values.Length * 2];
        for (var i = 0; i < values.Length; i++)
        {
            var floatBits = (uint)BitConverter.SingleToInt32Bits(values[i]);
            var bf16Bits = (ushort)(floatBits >> 16);
            BinaryPrimitives.WriteUInt16LittleEndian(bytes.AsSpan(i * 2), bf16Bits);
        }
        return bytes;
    }

    #endregion
}
