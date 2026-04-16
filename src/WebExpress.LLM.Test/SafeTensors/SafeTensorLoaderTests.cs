using System.Buffers.Binary;
using System.Text;
using System.Text.Json;
using WebExpress.LLM.Model;
using WebExpress.LLM.SafeTensors;

namespace WebExpress.LLM.Test.SafeTensors;

public sealed class SafeTensorLoaderTests
{
    [Fact]
    public void Constructor_ShouldParseHeaderAndDiscoverTensors()
    {
        var bytes = CreateSafeTensorsFile(new Dictionary<string, (string dtype, long[] shape, float[] data)>
        {
            ["weight1"] = ("F32", [2, 3], [1f, 2, 3, 4, 5, 6]),
            ["weight2"] = ("F32", [3], [7f, 8, 9])
        });

        var weights = ModelWeights.FromByteArray(bytes);
        var loader = new SafeTensorLoader(weights);

        Assert.Contains("weight1", loader.TensorNames);
        Assert.Contains("weight2", loader.TensorNames);
        Assert.Equal(2, loader.TensorNames.Count);
    }

    [Fact]
    public void ContainsTensor_ShouldReturnCorrectly()
    {
        var bytes = CreateSafeTensorsFile(new Dictionary<string, (string dtype, long[] shape, float[] data)>
        {
            ["model.embed_tokens.weight"] = ("F32", [4, 2], [1f, 2, 3, 4, 5, 6, 7, 8])
        });

        var weights = ModelWeights.FromByteArray(bytes);
        var loader = new SafeTensorLoader(weights);

        Assert.True(loader.ContainsTensor("model.embed_tokens.weight"));
        Assert.False(loader.ContainsTensor("nonexistent"));
    }

    [Fact]
    public void GetMetadata_ShouldReturnCorrectInfo()
    {
        var bytes = CreateSafeTensorsFile(new Dictionary<string, (string dtype, long[] shape, float[] data)>
        {
            ["test_tensor"] = ("F32", [3, 4], new float[12])
        });

        var weights = ModelWeights.FromByteArray(bytes);
        var loader = new SafeTensorLoader(weights);

        var meta = loader.GetMetadata("test_tensor");

        Assert.Equal("test_tensor", meta.Name);
        Assert.Equal("F32", meta.Dtype);
        Assert.Equal(2, meta.Shape.Count);
        Assert.Equal(3, meta.Shape[0]);
        Assert.Equal(4, meta.Shape[1]);
        Assert.Equal(4, meta.BytesPerElement);
        Assert.Equal(12, meta.ElementCount);
    }

    [Fact]
    public void GetMetadata_NonexistentTensor_ShouldThrow()
    {
        var bytes = CreateSafeTensorsFile(new Dictionary<string, (string dtype, long[] shape, float[] data)>
        {
            ["existing"] = ("F32", [2], [1f, 2])
        });

        var weights = ModelWeights.FromByteArray(bytes);
        var loader = new SafeTensorLoader(weights);

        Assert.Throws<KeyNotFoundException>(() => loader.GetMetadata("missing"));
    }

    [Fact]
    public void LoadTensor_F32_ShouldLoadCorrectData()
    {
        var expected = new float[] { 1.5f, 2.5f, 3.5f, 4.5f, 5.5f, 6.5f };
        var bytes = CreateSafeTensorsFile(new Dictionary<string, (string dtype, long[] shape, float[] data)>
        {
            ["test"] = ("F32", [2, 3], expected)
        });

        var weights = ModelWeights.FromByteArray(bytes);
        var loader = new SafeTensorLoader(weights);

        var tensor = loader.LoadTensor("test");

        Assert.Equal(2, tensor.Shape[0]);
        Assert.Equal(3, tensor.Shape[1]);

        for (var i = 0; i < expected.Length; i++)
        {
            Assert.Equal(expected[i], tensor.Data[i], 1e-6f);
        }
    }

    [Fact]
    public void LoadTensor_BF16_ShouldConvertToFloat32()
    {
        // Create BF16 data manually
        // BF16 is the upper 16 bits of float32
        var floatValues = new float[] { 1.0f, -2.0f, 0.5f, 3.0f };
        var bf16Bytes = new byte[floatValues.Length * 2];

        for (var i = 0; i < floatValues.Length; i++)
        {
            var floatBits = BitConverter.SingleToInt32Bits(floatValues[i]);
            var bf16Bits = (ushort)((uint)floatBits >> 16);
            BinaryPrimitives.WriteUInt16LittleEndian(bf16Bytes.AsSpan(i * 2), bf16Bits);
        }

        var bytes = CreateSafeTensorsFileRaw(new Dictionary<string, (string dtype, long[] shape, byte[] data)>
        {
            ["bf16_tensor"] = ("BF16", [4], bf16Bytes)
        });

        var weights = ModelWeights.FromByteArray(bytes);
        var loader = new SafeTensorLoader(weights);

        var tensor = loader.LoadTensor("bf16_tensor");

        Assert.Equal(4, tensor.Shape[0]);
        Assert.Equal(1.0f, tensor[0], 1e-2f);
        Assert.Equal(-2.0f, tensor[1], 1e-2f);
        Assert.Equal(0.5f, tensor[2], 1e-2f);
        Assert.Equal(3.0f, tensor[3], 1e-2f);
    }

    [Fact]
    public void LoadTensor_MultipleTensors_ShouldLoadEachCorrectly()
    {
        var bytes = CreateSafeTensorsFile(new Dictionary<string, (string dtype, long[] shape, float[] data)>
        {
            ["first"] = ("F32", [2], [10f, 20]),
            ["second"] = ("F32", [3], [30f, 40, 50])
        });

        var weights = ModelWeights.FromByteArray(bytes);
        var loader = new SafeTensorLoader(weights);

        var first = loader.LoadTensor("first");
        var second = loader.LoadTensor("second");

        Assert.Equal(10.0f, first[0]);
        Assert.Equal(20.0f, first[1]);
        Assert.Equal(30.0f, second[0]);
        Assert.Equal(40.0f, second[1]);
        Assert.Equal(50.0f, second[2]);
    }

    [Fact]
    public void Constructor_TooSmallFile_ShouldThrow()
    {
        var weights = ModelWeights.FromByteArray([1, 2, 3]);
        Assert.Throws<InvalidDataException>(() => new SafeTensorLoader(weights));
    }

    [Fact]
    public void Constructor_InvalidHeaderLength_ShouldThrow()
    {
        // Header length = -1 (invalid)
        var bytes = new byte[16];
        BinaryPrimitives.WriteInt64LittleEndian(bytes, -1);

        var weights = ModelWeights.FromByteArray(bytes);
        Assert.Throws<InvalidDataException>(() => new SafeTensorLoader(weights));
    }

    [Fact]
    public void TensorMetadata_BytesPerElement_ShouldReturnCorrectValues()
    {
        Assert.Equal(4, new TensorMetadata { Dtype = "F32" }.BytesPerElement);
        Assert.Equal(2, new TensorMetadata { Dtype = "F16" }.BytesPerElement);
        Assert.Equal(2, new TensorMetadata { Dtype = "BF16" }.BytesPerElement);
        Assert.Equal(4, new TensorMetadata { Dtype = "I32" }.BytesPerElement);
        Assert.Equal(8, new TensorMetadata { Dtype = "I64" }.BytesPerElement);
        Assert.Equal(1, new TensorMetadata { Dtype = "U8" }.BytesPerElement);
    }

    [Fact]
    public void TensorMetadata_ElementCount_ShouldComputeCorrectly()
    {
        var meta = new TensorMetadata { Shape = new long[] { 3, 4, 5 } };
        Assert.Equal(60, meta.ElementCount);
    }

    [Fact]
    public void TensorMetadata_EmptyShape_ShouldReturnOne()
    {
        var meta = new TensorMetadata { Shape = Array.Empty<long>() };
        Assert.Equal(1, meta.ElementCount);
    }

    [Fact]
    public void TensorMetadata_UnsupportedDtype_ShouldThrow()
    {
        var meta = new TensorMetadata { Dtype = "UNKNOWN" };
        Assert.Throws<NotSupportedException>(() => meta.BytesPerElement);
    }

    // ---------------------------------------------------------------
    // Helper: creates a valid SafeTensors file from F32 tensors
    // ---------------------------------------------------------------
    private static byte[] CreateSafeTensorsFile(
        Dictionary<string, (string dtype, long[] shape, float[] data)> tensors)
    {
        var rawTensors = new Dictionary<string, (string dtype, long[] shape, byte[] data)>();

        foreach (var (name, (dtype, shape, data)) in tensors)
        {
            var rawData = new byte[data.Length * 4];

            for (var i = 0; i < data.Length; i++)
            {
                BinaryPrimitives.WriteSingleLittleEndian(rawData.AsSpan(i * 4), data[i]);
            }

            rawTensors[name] = (dtype, shape, rawData);
        }

        return CreateSafeTensorsFileRaw(rawTensors);
    }

    private static byte[] CreateSafeTensorsFileRaw(
        Dictionary<string, (string dtype, long[] shape, byte[] data)> tensors)
    {
        // Build header JSON
        var header = new Dictionary<string, object>();
        long currentOffset = 0;

        foreach (var (name, (dtype, shape, data)) in tensors)
        {
            var endOffset = currentOffset + data.Length;
            header[name] = new
            {
                dtype,
                shape,
                data_offsets = new long[] { currentOffset, endOffset }
            };
            currentOffset = endOffset;
        }

        var headerJson = JsonSerializer.Serialize(header);
        var headerBytes = Encoding.UTF8.GetBytes(headerJson);

        // Assemble the file: 8-byte header length + header + data
        var totalDataSize = tensors.Values.Sum(t => t.data.Length);
        var result = new byte[8 + headerBytes.Length + totalDataSize];

        BinaryPrimitives.WriteInt64LittleEndian(result, headerBytes.Length);
        Array.Copy(headerBytes, 0, result, 8, headerBytes.Length);

        var dataOffset = 8 + headerBytes.Length;

        foreach (var (_, (_, _, data)) in tensors)
        {
            Array.Copy(data, 0, result, dataOffset, data.Length);
            dataOffset += data.Length;
        }

        return result;
    }
}
