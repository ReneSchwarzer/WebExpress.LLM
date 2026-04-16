using WebExpress.LLM.SafeTensors;

namespace WebExpress.LLM.Test.SafeTensors;

public sealed class SafeTensorIndexTests
{
    [Fact]
    public void Parse_ShouldParseValidIndexJson()
    {
        var json = """
        {
            "metadata": {
                "total_parameters": 26544131376,
                "total_size": 51611872412
            },
            "weight_map": {
                "model.embed_tokens.weight": "model-00001-of-00002.safetensors",
                "model.layers.0.self_attn.q_proj.weight": "model-00001-of-00002.safetensors",
                "model.layers.15.mlp.down_proj.weight": "model-00002-of-00002.safetensors",
                "model.norm.weight": "model-00002-of-00002.safetensors"
            }
        }
        """;

        var index = SafeTensorIndex.Parse(json);

        Assert.Equal(26544131376L, index.TotalParameters);
        Assert.Equal(51611872412L, index.TotalSize);
        Assert.Equal(4, index.WeightMap.Count);
        Assert.Equal(2, index.ShardFiles.Count);
        Assert.Contains("model-00001-of-00002.safetensors", index.ShardFiles);
        Assert.Contains("model-00002-of-00002.safetensors", index.ShardFiles);
    }

    [Fact]
    public void Parse_ShouldMapTensorsToCorrectShards()
    {
        var json = """
        {
            "metadata": {
                "total_parameters": 1000,
                "total_size": 2000
            },
            "weight_map": {
                "model.embed_vision.embedding_projection.weight": "model-00001-of-00002.safetensors",
                "model.language_model.embed_tokens.weight": "model-00001-of-00002.safetensors",
                "model.language_model.layers.0.experts.down_proj": "model-00001-of-00002.safetensors",
                "model.language_model.layers.20.experts.gate_up_proj": "model-00002-of-00002.safetensors",
                "model.language_model.lm_head.weight": "model-00002-of-00002.safetensors"
            }
        }
        """;

        var index = SafeTensorIndex.Parse(json);

        Assert.Equal("model-00001-of-00002.safetensors",
            index.WeightMap["model.embed_vision.embedding_projection.weight"]);
        Assert.Equal("model-00001-of-00002.safetensors",
            index.WeightMap["model.language_model.embed_tokens.weight"]);
        Assert.Equal("model-00001-of-00002.safetensors",
            index.WeightMap["model.language_model.layers.0.experts.down_proj"]);
        Assert.Equal("model-00002-of-00002.safetensors",
            index.WeightMap["model.language_model.layers.20.experts.gate_up_proj"]);
        Assert.Equal("model-00002-of-00002.safetensors",
            index.WeightMap["model.language_model.lm_head.weight"]);
    }

    [Fact]
    public void Parse_ShouldHandleMissingMetadata()
    {
        var json = """
        {
            "weight_map": {
                "tensor1": "shard-00001.safetensors"
            }
        }
        """;

        var index = SafeTensorIndex.Parse(json);

        Assert.Equal(0L, index.TotalParameters);
        Assert.Equal(0L, index.TotalSize);
        Assert.Single(index.WeightMap);
    }

    [Fact]
    public void Parse_ShouldHandlePartialMetadata()
    {
        var json = """
        {
            "metadata": {
                "total_parameters": 5000
            },
            "weight_map": {
                "tensor1": "shard-00001.safetensors"
            }
        }
        """;

        var index = SafeTensorIndex.Parse(json);

        Assert.Equal(5000L, index.TotalParameters);
        Assert.Equal(0L, index.TotalSize);
    }

    [Fact]
    public void Parse_ShouldCollectDistinctShardFiles()
    {
        var json = """
        {
            "metadata": {},
            "weight_map": {
                "tensor1": "shard-00001.safetensors",
                "tensor2": "shard-00001.safetensors",
                "tensor3": "shard-00002.safetensors",
                "tensor4": "shard-00003.safetensors",
                "tensor5": "shard-00003.safetensors"
            }
        }
        """;

        var index = SafeTensorIndex.Parse(json);

        Assert.Equal(5, index.WeightMap.Count);
        Assert.Equal(3, index.ShardFiles.Count);
        Assert.Contains("shard-00001.safetensors", index.ShardFiles);
        Assert.Contains("shard-00002.safetensors", index.ShardFiles);
        Assert.Contains("shard-00003.safetensors", index.ShardFiles);
    }

    [Fact]
    public void Parse_ShouldThrowWhenWeightMapIsMissing()
    {
        var json = """
        {
            "metadata": {
                "total_parameters": 1000
            }
        }
        """;

        Assert.Throws<InvalidDataException>(() => SafeTensorIndex.Parse(json));
    }

    [Fact]
    public void Parse_ShouldThrowWhenWeightMapIsEmpty()
    {
        var json = """
        {
            "metadata": {},
            "weight_map": {}
        }
        """;

        Assert.Throws<InvalidDataException>(() => SafeTensorIndex.Parse(json));
    }

    [Fact]
    public void Parse_ShouldThrowWhenJsonIsNull()
    {
        Assert.Throws<ArgumentException>(() => SafeTensorIndex.Parse(null));
    }

    [Fact]
    public void Parse_ShouldThrowWhenJsonIsEmpty()
    {
        Assert.Throws<ArgumentException>(() => SafeTensorIndex.Parse(""));
    }

    [Fact]
    public void Parse_ShouldThrowWhenJsonIsWhitespace()
    {
        Assert.Throws<ArgumentException>(() => SafeTensorIndex.Parse("   "));
    }

    [Fact]
    public void Parse_ShouldThrowWhenJsonIsMalformed()
    {
        Assert.Throws<InvalidDataException>(() => SafeTensorIndex.Parse("not valid json"));
    }

    [Fact]
    public void FromFile_ShouldThrowWhenFilePathIsNull()
    {
        Assert.Throws<ArgumentException>(() => SafeTensorIndex.FromFile(null));
    }

    [Fact]
    public void FromFile_ShouldThrowWhenFilePathIsEmpty()
    {
        Assert.Throws<ArgumentException>(() => SafeTensorIndex.FromFile(""));
    }

    [Fact]
    public void FromFile_ShouldThrowWhenFileDoesNotExist()
    {
        Assert.Throws<FileNotFoundException>(() =>
            SafeTensorIndex.FromFile("/nonexistent/path/index.json"));
    }

    [Fact]
    public void FromFile_ShouldParseFileOnDisk()
    {
        var tempPath = Path.Combine(Path.GetTempPath(), Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(tempPath);

        try
        {
            var indexJson = """
            {
                "metadata": {
                    "total_parameters": 100,
                    "total_size": 200
                },
                "weight_map": {
                    "layer.weight": "model-00001-of-00001.safetensors"
                }
            }
            """;

            var indexPath = Path.Combine(tempPath, SafeTensorIndex.DefaultFileName);
            File.WriteAllText(indexPath, indexJson);

            var index = SafeTensorIndex.FromFile(indexPath);

            Assert.Equal(100L, index.TotalParameters);
            Assert.Equal(200L, index.TotalSize);
            Assert.Single(index.WeightMap);
            Assert.Equal("model-00001-of-00001.safetensors", index.WeightMap["layer.weight"]);
        }
        finally
        {
            Directory.Delete(tempPath, recursive: true);
        }
    }

    [Fact]
    public void DefaultFileName_ShouldBeCorrect()
    {
        Assert.Equal("model.safetensors.index.json", SafeTensorIndex.DefaultFileName);
    }

    [Fact]
    public void Parse_ShouldHandleSingleShardWithManyTensors()
    {
        var json = """
        {
            "metadata": {
                "total_parameters": 500000,
                "total_size": 1000000
            },
            "weight_map": {
                "model.embed_tokens.weight": "model-00001-of-00001.safetensors",
                "model.layers.0.self_attn.q_proj.weight": "model-00001-of-00001.safetensors",
                "model.layers.0.self_attn.k_proj.weight": "model-00001-of-00001.safetensors",
                "model.layers.0.self_attn.v_proj.weight": "model-00001-of-00001.safetensors",
                "model.layers.0.mlp.gate_proj.weight": "model-00001-of-00001.safetensors",
                "model.norm.weight": "model-00001-of-00001.safetensors"
            }
        }
        """;

        var index = SafeTensorIndex.Parse(json);

        Assert.Equal(6, index.WeightMap.Count);
        Assert.Single(index.ShardFiles);
        Assert.Contains("model-00001-of-00001.safetensors", index.ShardFiles);
    }
}
