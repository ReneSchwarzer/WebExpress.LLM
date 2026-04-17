using System.Buffers.Binary;
using System.Text;
using System.Text.Json;
using WebExpress.LLM.Model;
using WebExpress.LLM.SafeTensors;

namespace WebExpress.LLM.Test.SafeTensors;

public sealed class ShardedSafeTensorLoaderTests
{
    [Fact]
    public void Constructor_ShouldLoadFromMultipleShards()
    {
        var (index, shardLoaders) = CreateTwoShardSetup();

        var loader = new ShardedSafeTensorLoader(index, shardLoaders);

        Assert.Equal(4, loader.TensorNames.Count);
        Assert.True(loader.ContainsTensor("tensor_a"));
        Assert.True(loader.ContainsTensor("tensor_b"));
        Assert.True(loader.ContainsTensor("tensor_c"));
        Assert.True(loader.ContainsTensor("tensor_d"));
    }

    [Fact]
    public void ContainsTensor_ShouldReturnFalseForMissing()
    {
        var (index, shardLoaders) = CreateTwoShardSetup();
        var loader = new ShardedSafeTensorLoader(index, shardLoaders);

        Assert.False(loader.ContainsTensor("nonexistent"));
    }

    [Fact]
    public void LoadTensor_ShouldLoadFromCorrectShard()
    {
        var (index, shardLoaders) = CreateTwoShardSetup();
        var loader = new ShardedSafeTensorLoader(index, shardLoaders);

        // tensor_a is in shard 1
        var tensorA = loader.LoadTensor("tensor_a");
        Assert.Equal(1.0f, tensorA[0], 1e-6f);
        Assert.Equal(2.0f, tensorA[1], 1e-6f);

        // tensor_c is in shard 2
        var tensorC = loader.LoadTensor("tensor_c");
        Assert.Equal(5.0f, tensorC[0], 1e-6f);
        Assert.Equal(6.0f, tensorC[1], 1e-6f);
    }

    [Fact]
    public void LoadTensor_ShouldLoadAllTensorsFromBothShards()
    {
        var (index, shardLoaders) = CreateTwoShardSetup();
        var loader = new ShardedSafeTensorLoader(index, shardLoaders);

        var tensorA = loader.LoadTensor("tensor_a");
        var tensorB = loader.LoadTensor("tensor_b");
        var tensorC = loader.LoadTensor("tensor_c");
        var tensorD = loader.LoadTensor("tensor_d");

        Assert.Equal(new[] { 1.0f, 2.0f }, tensorA.Data);
        Assert.Equal(new[] { 3.0f, 4.0f }, tensorB.Data);
        Assert.Equal(new[] { 5.0f, 6.0f }, tensorC.Data);
        Assert.Equal(new[] { 7.0f, 8.0f, 9.0f }, tensorD.Data);
    }

    [Fact]
    public void GetMetadata_ShouldReturnCorrectMetadataFromEachShard()
    {
        var (index, shardLoaders) = CreateTwoShardSetup();
        var loader = new ShardedSafeTensorLoader(index, shardLoaders);

        var metaA = loader.GetMetadata("tensor_a");
        Assert.Equal("tensor_a", metaA.Name);
        Assert.Equal("F32", metaA.Dtype);
        Assert.Equal(2L, metaA.Shape[0]);

        var metaD = loader.GetMetadata("tensor_d");
        Assert.Equal("tensor_d", metaD.Name);
        Assert.Equal("F32", metaD.Dtype);
        Assert.Equal(3L, metaD.Shape[0]);
    }

    [Fact]
    public void GetMetadata_ShouldThrowForMissingTensor()
    {
        var (index, shardLoaders) = CreateTwoShardSetup();
        var loader = new ShardedSafeTensorLoader(index, shardLoaders);

        Assert.Throws<KeyNotFoundException>(() => loader.GetMetadata("nonexistent"));
    }

    [Fact]
    public void LoadTensor_ShouldThrowForMissingTensor()
    {
        var (index, shardLoaders) = CreateTwoShardSetup();
        var loader = new ShardedSafeTensorLoader(index, shardLoaders);

        Assert.Throws<KeyNotFoundException>(() => loader.LoadTensor("nonexistent"));
    }

    [Fact]
    public void TensorNames_ShouldMatchIndexWeightMap()
    {
        var (index, shardLoaders) = CreateTwoShardSetup();
        var loader = new ShardedSafeTensorLoader(index, shardLoaders);

        Assert.Equal(index.WeightMap.Keys.OrderBy(k => k), loader.TensorNames.OrderBy(k => k));
    }

    [Fact]
    public void Index_ShouldReturnParsedIndex()
    {
        var (index, shardLoaders) = CreateTwoShardSetup();
        var loader = new ShardedSafeTensorLoader(index, shardLoaders);

        Assert.Same(index, loader.Index);
    }

    [Fact]
    public void Constructor_ShouldThrowWhenIndexIsNull()
    {
        Assert.Throws<ArgumentNullException>(() =>
            new ShardedSafeTensorLoader(null, new Dictionary<string, SafeTensorLoader>()));
    }

    [Fact]
    public void Constructor_ShouldThrowWhenShardLoadersIsNull()
    {
        var index = SafeTensorIndex.Parse("""
        {
            "metadata": {},
            "weight_map": { "t": "shard.safetensors" }
        }
        """);

        Assert.Throws<ArgumentNullException>(() =>
            new ShardedSafeTensorLoader(index, (Dictionary<string, SafeTensorLoader>)null!));
    }

    [Fact]
    public void Constructor_ShouldThrowWhenShardLoaderMissing()
    {
        var index = SafeTensorIndex.Parse("""
        {
            "metadata": {},
            "weight_map": { "tensor": "missing-shard.safetensors" }
        }
        """);

        var emptyLoaders = new Dictionary<string, SafeTensorLoader>();

        Assert.Throws<InvalidDataException>(() =>
            new ShardedSafeTensorLoader(index, emptyLoaders));
    }

    [Fact]
    public void Constructor_WithDirectory_ShouldThrowWhenDirectoryIsNull()
    {
        var index = SafeTensorIndex.Parse("""
        {
            "metadata": {},
            "weight_map": { "t": "shard.safetensors" }
        }
        """);

        Assert.Throws<ArgumentException>(() =>
            new ShardedSafeTensorLoader(index, (string)null!));
    }

    [Fact]
    public void Constructor_WithDirectory_ShouldThrowWhenShardFileMissing()
    {
        var tempPath = Path.Combine(Path.GetTempPath(), Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(tempPath);

        try
        {
            var index = SafeTensorIndex.Parse("""
            {
                "metadata": {},
                "weight_map": { "tensor": "nonexistent-shard.safetensors" }
            }
            """);

            Assert.Throws<FileNotFoundException>(() =>
                new ShardedSafeTensorLoader(index, tempPath));
        }
        finally
        {
            Directory.Delete(tempPath, recursive: true);
        }
    }

    [Fact]
    public void Constructor_WithDirectory_ShouldLoadShardFilesFromDisk()
    {
        var tempPath = Path.Combine(Path.GetTempPath(), Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(tempPath);

        try
        {
            // Create shard files on disk
            var shard1Data = CreateSafeTensorsFile(new Dictionary<string, (string dtype, long[] shape, float[] data)>
            {
                ["embed.weight"] = ("F32", [2, 2], [1f, 2, 3, 4])
            });
            File.WriteAllBytes(Path.Combine(tempPath, "model-00001-of-00002.safetensors"), shard1Data);

            var shard2Data = CreateSafeTensorsFile(new Dictionary<string, (string dtype, long[] shape, float[] data)>
            {
                ["lm_head.weight"] = ("F32", [2], [5f, 6])
            });
            File.WriteAllBytes(Path.Combine(tempPath, "model-00002-of-00002.safetensors"), shard2Data);

            var index = SafeTensorIndex.Parse("""
            {
                "metadata": {
                    "total_parameters": 6,
                    "total_size": 24
                },
                "weight_map": {
                    "embed.weight": "model-00001-of-00002.safetensors",
                    "lm_head.weight": "model-00002-of-00002.safetensors"
                }
            }
            """);

            using var loader = new ShardedSafeTensorLoader(index, tempPath);

            Assert.Equal(2, loader.TensorNames.Count);
            Assert.True(loader.ContainsTensor("embed.weight"));
            Assert.True(loader.ContainsTensor("lm_head.weight"));

            var embedTensor = loader.LoadTensor("embed.weight");
            Assert.Equal(2, embedTensor.Shape[0]);
            Assert.Equal(2, embedTensor.Shape[1]);
            Assert.Equal(1.0f, embedTensor.Data[0], 1e-6f);
            Assert.Equal(4.0f, embedTensor.Data[3], 1e-6f);

            var lmHeadTensor = loader.LoadTensor("lm_head.weight");
            Assert.Equal(2, lmHeadTensor.Shape[0]);
            Assert.Equal(5.0f, lmHeadTensor.Data[0], 1e-6f);
            Assert.Equal(6.0f, lmHeadTensor.Data[1], 1e-6f);
        }
        finally
        {
            Directory.Delete(tempPath, recursive: true);
        }
    }

    [Fact]
    public void ISafeTensorLoader_ShouldBeUsableAsInterface()
    {
        var (index, shardLoaders) = CreateTwoShardSetup();
        ISafeTensorLoader loader = new ShardedSafeTensorLoader(index, shardLoaders);

        Assert.True(loader.ContainsTensor("tensor_a"));
        Assert.Equal(4, loader.TensorNames.Count);

        var tensor = loader.LoadTensor("tensor_b");
        Assert.Equal(3.0f, tensor[0], 1e-6f);
    }

    [Fact]
    public void ThreeShards_ShouldDistributeTensorsCorrectly()
    {
        // Create three shard files in memory
        var shard1Bytes = CreateSafeTensorsFile(new Dictionary<string, (string dtype, long[] shape, float[] data)>
        {
            ["layer.0.weight"] = ("F32", [2], [10f, 20])
        });
        var shard2Bytes = CreateSafeTensorsFile(new Dictionary<string, (string dtype, long[] shape, float[] data)>
        {
            ["layer.1.weight"] = ("F32", [2], [30f, 40])
        });
        var shard3Bytes = CreateSafeTensorsFile(new Dictionary<string, (string dtype, long[] shape, float[] data)>
        {
            ["layer.2.weight"] = ("F32", [2], [50f, 60])
        });

        var index = SafeTensorIndex.Parse("""
        {
            "metadata": {
                "total_parameters": 6,
                "total_size": 24
            },
            "weight_map": {
                "layer.0.weight": "model-00001-of-00003.safetensors",
                "layer.1.weight": "model-00002-of-00003.safetensors",
                "layer.2.weight": "model-00003-of-00003.safetensors"
            }
        }
        """);

        var shardLoaders = new Dictionary<string, SafeTensorLoader>
        {
            ["model-00001-of-00003.safetensors"] = new(ModelWeights.FromByteArray(shard1Bytes)),
            ["model-00002-of-00003.safetensors"] = new(ModelWeights.FromByteArray(shard2Bytes)),
            ["model-00003-of-00003.safetensors"] = new(ModelWeights.FromByteArray(shard3Bytes))
        };

        var loader = new ShardedSafeTensorLoader(index, shardLoaders);

        Assert.Equal(3, loader.TensorNames.Count);
        Assert.Equal(3, loader.Index.ShardFiles.Count);

        Assert.Equal(new[] { 10f, 20f }, loader.LoadTensor("layer.0.weight").Data);
        Assert.Equal(new[] { 30f, 40f }, loader.LoadTensor("layer.1.weight").Data);
        Assert.Equal(new[] { 50f, 60f }, loader.LoadTensor("layer.2.weight").Data);
    }

    [Fact]
    public void LoadTensor_WithGlobalOffsetsAndCrossShardMetadata_ShouldLoadCorrectly()
    {
        // Reproduce the bug: each shard's header lists ALL model tensors with
        // global offsets, but only stores data for its own tensors. The index
        // correctly maps tensors to shards. Without the ShardedSafeTensorLoader
        // recomputing the base offset per shard (using only index-mapped tensors),
        // the SafeTensorLoader's ComputeBaseOffset picks up offsets from other
        // shards' tensors, producing a wrong base and an out-of-range read.
        var shard1Data = new float[] { 1f, 2f, 3f, 4f };    // 16 bytes
        var shard2Data = new float[] { 10f, 20f, 30f };      // 12 bytes

        var shard1Raw = FloatsToBytes(shard1Data);
        var shard2Raw = FloatsToBytes(shard2Data);

        // Global offsets: shard 1 tensors at [0..8] and [8..16], shard 2 at [16..24] and [24..28]
        long globalBase2 = shard1Raw.Length; // = 16

        // Shard 1 file: header lists ALL tensors with global offsets, data has shard1 only
        var shard1Bytes = CreateSafeTensorsFileWithGlobalHeader(
            allTensorHeaders: new Dictionary<string, (string dtype, long[] shape, long begin, long end)>
            {
                ["tensor_a"] = ("F32", [2], 0, 8),
                ["tensor_b"] = ("F32", [2], 8, 16),
                ["tensor_c"] = ("F32", [2], globalBase2, globalBase2 + 8),
                ["tensor_d"] = ("F32", [1], globalBase2 + 8, globalBase2 + 12)
            },
            localData: shard1Raw);

        // Shard 2 file: header lists ALL tensors with global offsets, data has shard2 only
        var shard2Bytes = CreateSafeTensorsFileWithGlobalHeader(
            allTensorHeaders: new Dictionary<string, (string dtype, long[] shape, long begin, long end)>
            {
                ["tensor_a"] = ("F32", [2], 0, 8),
                ["tensor_b"] = ("F32", [2], 8, 16),
                ["tensor_c"] = ("F32", [2], globalBase2, globalBase2 + 8),
                ["tensor_d"] = ("F32", [1], globalBase2 + 8, globalBase2 + 12)
            },
            localData: shard2Raw);

        var index = SafeTensorIndex.Parse("""
        {
            "metadata": { "total_parameters": 7, "total_size": 28 },
            "weight_map": {
                "tensor_a": "shard-00001.safetensors",
                "tensor_b": "shard-00001.safetensors",
                "tensor_c": "shard-00002.safetensors",
                "tensor_d": "shard-00002.safetensors"
            }
        }
        """);

        var shardLoaders = new Dictionary<string, SafeTensorLoader>
        {
            ["shard-00001.safetensors"] = new(ModelWeights.FromByteArray(shard1Bytes)),
            ["shard-00002.safetensors"] = new(ModelWeights.FromByteArray(shard2Bytes))
        };

        var loader = new ShardedSafeTensorLoader(index, shardLoaders);

        // Load from shard 1 (local offsets start at 0, should work regardless)
        var tA = loader.LoadTensor("tensor_a");
        Assert.Equal(new[] { 1f, 2f }, tA.Data);

        var tB = loader.LoadTensor("tensor_b");
        Assert.Equal(new[] { 3f, 4f }, tB.Data);

        // Load from shard 2 — these have global offsets > shard2 file size.
        // Without the fix, this would throw ArgumentOutOfRangeException because
        // the base offset would be 0 (from tensor_a's global offset in the header).
        var tC = loader.LoadTensor("tensor_c");
        Assert.Equal(new[] { 10f, 20f }, tC.Data);

        var tD = loader.LoadTensor("tensor_d");
        Assert.Equal(new[] { 30f }, tD.Data);
    }

    // ---------------------------------------------------------------
    // Helpers
    // ---------------------------------------------------------------

    private static (SafeTensorIndex index, Dictionary<string, SafeTensorLoader> shardLoaders) CreateTwoShardSetup()
    {
        var shard1Bytes = CreateSafeTensorsFile(new Dictionary<string, (string dtype, long[] shape, float[] data)>
        {
            ["tensor_a"] = ("F32", [2], [1f, 2]),
            ["tensor_b"] = ("F32", [2], [3f, 4])
        });

        var shard2Bytes = CreateSafeTensorsFile(new Dictionary<string, (string dtype, long[] shape, float[] data)>
        {
            ["tensor_c"] = ("F32", [2], [5f, 6]),
            ["tensor_d"] = ("F32", [3], [7f, 8, 9])
        });

        var index = SafeTensorIndex.Parse("""
        {
            "metadata": {
                "total_parameters": 9,
                "total_size": 36
            },
            "weight_map": {
                "tensor_a": "shard-00001.safetensors",
                "tensor_b": "shard-00001.safetensors",
                "tensor_c": "shard-00002.safetensors",
                "tensor_d": "shard-00002.safetensors"
            }
        }
        """);

        var shardLoaders = new Dictionary<string, SafeTensorLoader>
        {
            ["shard-00001.safetensors"] = new(ModelWeights.FromByteArray(shard1Bytes)),
            ["shard-00002.safetensors"] = new(ModelWeights.FromByteArray(shard2Bytes))
        };

        return (index, shardLoaders);
    }

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

    /// <summary>
    /// Creates a SafeTensors file where the header lists ALL tensors (possibly from
    /// other shards) with arbitrary data offsets, but only contains the given local data.
    /// </summary>
    private static byte[] CreateSafeTensorsFileWithGlobalHeader(
        Dictionary<string, (string dtype, long[] shape, long begin, long end)> allTensorHeaders,
        byte[] localData)
    {
        var header = new Dictionary<string, object>();

        foreach (var (name, (dtype, shape, begin, end)) in allTensorHeaders)
        {
            header[name] = new
            {
                dtype,
                shape,
                data_offsets = new long[] { begin, end }
            };
        }

        var headerJson = JsonSerializer.Serialize(header);
        var headerBytes = Encoding.UTF8.GetBytes(headerJson);

        var result = new byte[8 + headerBytes.Length + localData.Length];

        BinaryPrimitives.WriteInt64LittleEndian(result, headerBytes.Length);
        Array.Copy(headerBytes, 0, result, 8, headerBytes.Length);
        Array.Copy(localData, 0, result, 8 + headerBytes.Length, localData.Length);

        return result;
    }

    private static byte[] FloatsToBytes(float[] values)
    {
        var bytes = new byte[values.Length * 4];

        for (var i = 0; i < values.Length; i++)
        {
            BinaryPrimitives.WriteSingleLittleEndian(bytes.AsSpan(i * 4), values[i]);
        }

        return bytes;
    }
}
