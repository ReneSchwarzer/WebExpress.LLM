using System.Buffers.Binary;
using System.Text;
using System.Text.Json;
using System.Text.Json.Nodes;
using WebExpress.LLM.Model;
using WebExpress.LLM.SafeTensors;

namespace WebExpress.LLM.Test.Model;

public sealed class ModelLoaderTests
{
    [Fact]
    public void Load_ShouldReadConfigurationAndWeights()
    {
        var tempPath = Path.Combine(Path.GetTempPath(), Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(tempPath);

        try
        {
            var configuration = new ModelConfiguration
            {
                ModelName = "gemma-4-mini",
                VocabularySize = 256000,
                ContextLength = 8192,
                HiddenSize = 2048,
                IntermediateSize = 8192,
                NumberOfLayers = 18,
                NumberOfAttentionHeads = 8,
                NumberOfKeyValueHeads = 1,
                HeadDimension = 256
            };

            File.WriteAllText(
                Path.Combine(tempPath, ModelLoader.DefaultConfigurationFileName),
                JsonSerializer.Serialize(configuration));
            File.WriteAllBytes(Path.Combine(tempPath, ModelLoader.DefaultWeightsFileName), [1, 2, 3, 4]);

            var loader = new ModelLoader();
            var model = loader.Load(tempPath);

            Assert.Equal("gemma-4-mini", model.Configuration.ModelName);
            Assert.Equal(256000, model.Configuration.VocabularySize);
            Assert.Equal(8192, model.Configuration.ContextLength);
            Assert.Equal(2048, model.Configuration.HiddenSize);
            Assert.Equal(18, model.Configuration.NumberOfLayers);
            Assert.Equal([1, 2, 3, 4], model.Weights.ToByteArray());

            model.Dispose();
        }
        finally
        {
            Directory.Delete(tempPath, recursive: true);
        }
    }

    [Fact]
    public void Load_ShouldReadSafetensorsWeights()
    {
        var tempPath = Path.Combine(Path.GetTempPath(), Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(tempPath);

        try
        {
            var configuration = new ModelConfiguration
            {
                ModelName = "test-model",
                VocabularySize = 32000,
                ContextLength = 2048,
                HiddenSize = 512,
                IntermediateSize = 2048,
                NumberOfLayers = 6,
                NumberOfAttentionHeads = 8,
                NumberOfKeyValueHeads = 1,
                HeadDimension = 64
            };

            File.WriteAllText(
                Path.Combine(tempPath, ModelLoader.DefaultConfigurationFileName),
                JsonSerializer.Serialize(configuration));
            File.WriteAllBytes(Path.Combine(tempPath, "model.safetensors"), [5, 6, 7, 8]);

            var loader = new ModelLoader();
            var model = loader.Load(tempPath);

            Assert.Equal("test-model", model.Configuration.ModelName);
            Assert.Equal([5, 6, 7, 8], model.Weights.ToByteArray());

            model.Dispose();
        }
        finally
        {
            Directory.Delete(tempPath, recursive: true);
        }
    }

    [Fact]
    public void Load_ShouldPreferSafetensorsOverOtherFormats()
    {
        var tempPath = Path.Combine(Path.GetTempPath(), Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(tempPath);

        try
        {
            var configuration = new ModelConfiguration
            {
                ModelName = "multi-format-model",
                VocabularySize = 32000,
                ContextLength = 2048,
                HiddenSize = 512,
                IntermediateSize = 2048,
                NumberOfLayers = 6,
                NumberOfAttentionHeads = 8,
                NumberOfKeyValueHeads = 1,
                HeadDimension = 64
            };

            File.WriteAllText(
                Path.Combine(tempPath, ModelLoader.DefaultConfigurationFileName),
                JsonSerializer.Serialize(configuration));

            // Create multiple weight files - safetensors should be preferred
            File.WriteAllBytes(Path.Combine(tempPath, "model.safetensors"), [1, 2, 3]);
            File.WriteAllBytes(Path.Combine(tempPath, ModelLoader.DefaultWeightsFileName), [4, 5, 6]);
            File.WriteAllBytes(Path.Combine(tempPath, "pytorch_model.bin"), [7, 8, 9]);

            var loader = new ModelLoader();
            var model = loader.Load(tempPath);

            // Should load safetensors file (first in priority list)
            Assert.Equal([1, 2, 3], model.Weights.ToByteArray());

            model.Dispose();
        }
        finally
        {
            Directory.Delete(tempPath, recursive: true);
        }
    }

    [Fact]
    public void Load_ShouldReadGemma4NestedConfiguration()
    {
        var tempPath = Path.Combine(Path.GetTempPath(), Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(tempPath);

        try
        {
            // Gemma-4 style config: critical text parameters are inside text_config, not at the top level.
            var gemma4Config = new JsonObject
            {
                ["model_type"] = "gemma4",
                ["architectures"] = new JsonArray("Gemma4ForConditionalGeneration"),
                ["dtype"] = "bfloat16",
                ["tie_word_embeddings"] = true,
                ["audio_token_id"] = 258881,
                ["boa_token_id"] = 256000,
                ["boi_token_id"] = 255999,
                ["eoa_token_id"] = 258883,
                ["eoa_token_index"] = 258883,
                ["eoi_token_id"] = 258882,
                ["image_token_id"] = 258880,
                ["video_token_id"] = 258884,
                ["vision_soft_tokens_per_image"] = 280,
                ["text_config"] = new JsonObject
                {
                    ["model_type"] = "gemma4_text",
                    ["vocab_size"] = 262144,
                    ["max_position_embeddings"] = 131072,
                    ["hidden_size"] = 1536,
                    ["intermediate_size"] = 6144,
                    ["num_hidden_layers"] = 35,
                    ["num_attention_heads"] = 8,
                    ["num_key_value_heads"] = 1,
                    ["head_dim"] = 256,
                    ["global_head_dim"] = 512,
                    ["rms_norm_eps"] = 1e-6,
                    ["sliding_window"] = 512,
                    ["rope_parameters"] = new JsonObject
                    {
                        ["full_attention"] = new JsonObject
                        {
                            ["rope_theta"] = 1000000.0,
                            ["rope_type"] = "proportional",
                            ["partial_rotary_factor"] = 0.25
                        },
                        ["sliding_attention"] = new JsonObject
                        {
                            ["rope_theta"] = 10000.0,
                            ["rope_type"] = "default"
                        }
                    }
                },
                ["audio_config"] = new JsonObject
                {
                    ["model_type"] = "gemma4_audio",
                    ["hidden_size"] = 1024,
                    ["num_hidden_layers"] = 12,
                    ["num_attention_heads"] = 8,
                    ["output_proj_dims"] = 1536,
                    ["rms_norm_eps"] = 1e-6,
                    ["subsampling_conv_channels"] = new JsonArray(128, 32)
                },
                ["vision_config"] = new JsonObject
                {
                    ["model_type"] = "gemma4_vision",
                    ["hidden_size"] = 768,
                    ["num_hidden_layers"] = 16,
                    ["num_attention_heads"] = 12,
                    ["num_key_value_heads"] = 12,
                    ["intermediate_size"] = 3072,
                    ["head_dim"] = 64,
                    ["patch_size"] = 16,
                    ["default_output_length"] = 280,
                    ["rms_norm_eps"] = 1e-6,
                    ["rope_parameters"] = new JsonObject
                    {
                        ["rope_theta"] = 100.0,
                        ["rope_type"] = "default"
                    }
                }
            };

            File.WriteAllText(
                Path.Combine(tempPath, ModelLoader.DefaultConfigurationFileName),
                gemma4Config.ToJsonString());
            File.WriteAllBytes(Path.Combine(tempPath, ModelLoader.DefaultWeightsFileName), [1, 2, 3, 4]);

            var loader = new ModelLoader();
            var model = loader.Load(tempPath);

            // Flat properties should resolve from text_config
            Assert.Equal(262144, model.Configuration.VocabularySize);
            Assert.Equal(131072, model.Configuration.ContextLength);
            Assert.Equal(1536, model.Configuration.HiddenSize);
            Assert.Equal(6144, model.Configuration.IntermediateSize);
            Assert.Equal(35, model.Configuration.NumberOfLayers);
            Assert.Equal(8, model.Configuration.NumberOfAttentionHeads);
            Assert.Equal(1, model.Configuration.NumberOfKeyValueHeads);
            Assert.Equal(256, model.Configuration.HeadDimension);
            Assert.Equal(10000.0f, model.Configuration.RopeTheta);

            // Top-level multi-modal token IDs
            Assert.Equal("gemma4", model.Configuration.ModelType);
            Assert.Equal(258881, model.Configuration.AudioTokenId);
            Assert.Equal(256000, model.Configuration.BoaTokenId);
            Assert.Equal(255999, model.Configuration.BoiTokenId);
            Assert.Equal(258883, model.Configuration.EoaTokenId);
            Assert.Equal(258882, model.Configuration.EoiTokenId);
            Assert.Equal(258880, model.Configuration.ImageTokenId);
            Assert.Equal(258884, model.Configuration.VideoTokenId);
            Assert.Equal(280, model.Configuration.VisionSoftTokensPerImage);

            // Nested sub-configs
            Assert.NotNull(model.Configuration.TextConfig);
            Assert.Equal("gemma4_text", model.Configuration.TextConfig.ModelType);
            Assert.Equal(512, model.Configuration.TextConfig.SlidingWindow);
            Assert.NotNull(model.Configuration.TextConfig.RopeParameters?.SlidingAttention);
            Assert.Equal(10000.0f, model.Configuration.TextConfig.RopeParameters.SlidingAttention.RopeTheta);
            Assert.NotNull(model.Configuration.TextConfig.RopeParameters.FullAttention);
            Assert.Equal(1000000.0f, model.Configuration.TextConfig.RopeParameters.FullAttention.RopeTheta);

            Assert.NotNull(model.Configuration.AudioConfig);
            Assert.Equal("gemma4_audio", model.Configuration.AudioConfig.ModelType);
            Assert.Equal(1024, model.Configuration.AudioConfig.HiddenSize);
            Assert.Equal(2, model.Configuration.AudioConfig.SubsamplingConvChannels.Count);

            Assert.NotNull(model.Configuration.VisionConfig);
            Assert.Equal("gemma4_vision", model.Configuration.VisionConfig.ModelType);
            Assert.Equal(768, model.Configuration.VisionConfig.HiddenSize);
            Assert.Equal(100.0f, model.Configuration.VisionConfig.RopeParameters.RopeTheta);

            model.Dispose();
        }
        finally
        {
            Directory.Delete(tempPath, recursive: true);
        }
    }

    [Fact]
    public void Load_ShouldThrowWhenNoWeightsFileExists()
    {
        var tempPath = Path.Combine(Path.GetTempPath(), Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(tempPath);

        try
        {
            var configuration = new ModelConfiguration
            {
                ModelName = "test-model",
                VocabularySize = 32000,
                ContextLength = 2048,
                HiddenSize = 512,
                IntermediateSize = 2048,
                NumberOfLayers = 6,
                NumberOfAttentionHeads = 8,
                NumberOfKeyValueHeads = 1,
                HeadDimension = 64
            };

            File.WriteAllText(
                Path.Combine(tempPath, ModelLoader.DefaultConfigurationFileName),
                JsonSerializer.Serialize(configuration));

            var loader = new ModelLoader();
            var exception = Assert.Throws<FileNotFoundException>(() => loader.Load(tempPath));

            Assert.Contains("Model weights file was not found", exception.Message);
            Assert.Contains("model.safetensors", exception.Message);
        }
        finally
        {
            Directory.Delete(tempPath, recursive: true);
        }
    }

    [Fact]
    public void Load_ShouldThrowWhenVocabularySizeIsZero()
    {
        var tempPath = Path.Combine(Path.GetTempPath(), Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(tempPath);

        try
        {
            var configuration = new ModelConfiguration
            {
                ModelName = "test-model",
                VocabularySize = 0,
                ContextLength = 2048,
                HiddenSize = 512,
                IntermediateSize = 2048,
                NumberOfLayers = 6,
                NumberOfAttentionHeads = 8,
                NumberOfKeyValueHeads = 1,
                HeadDimension = 64
            };

            File.WriteAllText(
                Path.Combine(tempPath, ModelLoader.DefaultConfigurationFileName),
                JsonSerializer.Serialize(configuration));
            File.WriteAllBytes(Path.Combine(tempPath, ModelLoader.DefaultWeightsFileName), [1, 2, 3, 4]);

            var loader = new ModelLoader();
            var exception = Assert.Throws<InvalidDataException>(() => loader.Load(tempPath));

            Assert.Contains("invalid vocabulary size", exception.Message);
            Assert.Contains("must be greater than zero", exception.Message);
        }
        finally
        {
            Directory.Delete(tempPath, recursive: true);
        }
    }

    [Fact]
    public void Load_ShouldThrowWhenContextLengthIsZero()
    {
        var tempPath = Path.Combine(Path.GetTempPath(), Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(tempPath);

        try
        {
            var configuration = new ModelConfiguration
            {
                ModelName = "test-model",
                VocabularySize = 32000,
                ContextLength = 0,
                HiddenSize = 512,
                IntermediateSize = 2048,
                NumberOfLayers = 6,
                NumberOfAttentionHeads = 8,
                NumberOfKeyValueHeads = 1,
                HeadDimension = 64
            };

            File.WriteAllText(
                Path.Combine(tempPath, ModelLoader.DefaultConfigurationFileName),
                JsonSerializer.Serialize(configuration));
            File.WriteAllBytes(Path.Combine(tempPath, ModelLoader.DefaultWeightsFileName), [1, 2, 3, 4]);

            var loader = new ModelLoader();
            var exception = Assert.Throws<InvalidDataException>(() => loader.Load(tempPath));

            Assert.Contains("invalid context length", exception.Message);
            Assert.Contains("must be greater than zero", exception.Message);
        }
        finally
        {
            Directory.Delete(tempPath, recursive: true);
        }
    }

    [Fact]
    public void Load_ShouldDetectShardedWeightsFromIndexFile()
    {
        var tempPath = Path.Combine(Path.GetTempPath(), Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(tempPath);

        try
        {
            WriteValidConfiguration(tempPath);

            // Create shard files
            var shard1 = CreateSafeTensorsFile(new Dictionary<string, (string dtype, long[] shape, float[] data)>
            {
                ["model.embed_tokens.weight"] = ("F32", [2, 2], [1f, 2, 3, 4])
            });
            File.WriteAllBytes(Path.Combine(tempPath, "model-00001-of-00002.safetensors"), shard1);

            var shard2 = CreateSafeTensorsFile(new Dictionary<string, (string dtype, long[] shape, float[] data)>
            {
                ["model.norm.weight"] = ("F32", [2], [5f, 6])
            });
            File.WriteAllBytes(Path.Combine(tempPath, "model-00002-of-00002.safetensors"), shard2);

            // Create index file
            var indexJson = """
            {
                "metadata": {
                    "total_parameters": 6,
                    "total_size": 24
                },
                "weight_map": {
                    "model.embed_tokens.weight": "model-00001-of-00002.safetensors",
                    "model.norm.weight": "model-00002-of-00002.safetensors"
                }
            }
            """;
            File.WriteAllText(
                Path.Combine(tempPath, SafeTensorIndex.DefaultFileName),
                indexJson);

            var loader = new ModelLoader();
            var model = loader.Load(tempPath);

            Assert.NotNull(model.ShardedLoader);
            Assert.Null(model.Weights);
            Assert.True(model.ShardedLoader.ContainsTensor("model.embed_tokens.weight"));
            Assert.True(model.ShardedLoader.ContainsTensor("model.norm.weight"));
            Assert.Equal(2, model.ShardedLoader.TensorNames.Count);

            model.Dispose();
        }
        finally
        {
            Directory.Delete(tempPath, recursive: true);
        }
    }

    [Fact]
    public void Load_ShardedWeights_ShouldLoadTensorsCorrectly()
    {
        var tempPath = Path.Combine(Path.GetTempPath(), Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(tempPath);

        try
        {
            WriteValidConfiguration(tempPath);

            var shard1 = CreateSafeTensorsFile(new Dictionary<string, (string dtype, long[] shape, float[] data)>
            {
                ["weight_a"] = ("F32", [3], [10f, 20, 30])
            });
            File.WriteAllBytes(Path.Combine(tempPath, "model-00001-of-00002.safetensors"), shard1);

            var shard2 = CreateSafeTensorsFile(new Dictionary<string, (string dtype, long[] shape, float[] data)>
            {
                ["weight_b"] = ("F32", [2], [40f, 50])
            });
            File.WriteAllBytes(Path.Combine(tempPath, "model-00002-of-00002.safetensors"), shard2);

            var indexJson = """
            {
                "metadata": {
                    "total_parameters": 5,
                    "total_size": 20
                },
                "weight_map": {
                    "weight_a": "model-00001-of-00002.safetensors",
                    "weight_b": "model-00002-of-00002.safetensors"
                }
            }
            """;
            File.WriteAllText(
                Path.Combine(tempPath, SafeTensorIndex.DefaultFileName),
                indexJson);

            var loader = new ModelLoader();
            var model = loader.Load(tempPath);

            var tensorA = model.ShardedLoader.LoadTensor("weight_a");
            Assert.Equal(new[] { 10f, 20f, 30f }, tensorA.Data);

            var tensorB = model.ShardedLoader.LoadTensor("weight_b");
            Assert.Equal(new[] { 40f, 50f }, tensorB.Data);

            model.Dispose();
        }
        finally
        {
            Directory.Delete(tempPath, recursive: true);
        }
    }

    [Fact]
    public void Load_ShouldPreferIndexFileOverSingleSafetensors()
    {
        var tempPath = Path.Combine(Path.GetTempPath(), Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(tempPath);

        try
        {
            WriteValidConfiguration(tempPath);

            // Create both a single safetensors file and sharded files with index
            File.WriteAllBytes(Path.Combine(tempPath, "model.safetensors"), [1, 2, 3, 4, 5, 6, 7, 8]);

            var shard1 = CreateSafeTensorsFile(new Dictionary<string, (string dtype, long[] shape, float[] data)>
            {
                ["tensor_from_shard"] = ("F32", [2], [100f, 200])
            });
            File.WriteAllBytes(Path.Combine(tempPath, "model-00001-of-00001.safetensors"), shard1);

            var indexJson = """
            {
                "metadata": {},
                "weight_map": {
                    "tensor_from_shard": "model-00001-of-00001.safetensors"
                }
            }
            """;
            File.WriteAllText(
                Path.Combine(tempPath, SafeTensorIndex.DefaultFileName),
                indexJson);

            var loader = new ModelLoader();
            var model = loader.Load(tempPath);

            // Sharded should be preferred when index file exists
            Assert.NotNull(model.ShardedLoader);
            Assert.Null(model.Weights);
            Assert.True(model.ShardedLoader.ContainsTensor("tensor_from_shard"));

            model.Dispose();
        }
        finally
        {
            Directory.Delete(tempPath, recursive: true);
        }
    }

    [Fact]
    public void Load_ShouldThrowWhenShardedIndexReferencesNonexistentShard()
    {
        var tempPath = Path.Combine(Path.GetTempPath(), Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(tempPath);

        try
        {
            WriteValidConfiguration(tempPath);

            var indexJson = """
            {
                "metadata": {},
                "weight_map": {
                    "tensor": "nonexistent-shard.safetensors"
                }
            }
            """;
            File.WriteAllText(
                Path.Combine(tempPath, SafeTensorIndex.DefaultFileName),
                indexJson);

            var loader = new ModelLoader();
            Assert.Throws<FileNotFoundException>(() => loader.Load(tempPath));
        }
        finally
        {
            Directory.Delete(tempPath, recursive: true);
        }
    }

    // ---------------------------------------------------------------
    // Helpers for sharded test cases
    // ---------------------------------------------------------------

    private static void WriteValidConfiguration(string tempPath)
    {
        var configuration = new ModelConfiguration
        {
            ModelName = "test-model",
            VocabularySize = 32000,
            ContextLength = 2048,
            HiddenSize = 512,
            IntermediateSize = 2048,
            NumberOfLayers = 6,
            NumberOfAttentionHeads = 8,
            NumberOfKeyValueHeads = 1,
            HeadDimension = 64
        };

        File.WriteAllText(
            Path.Combine(tempPath, ModelLoader.DefaultConfigurationFileName),
            JsonSerializer.Serialize(configuration));
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
}
