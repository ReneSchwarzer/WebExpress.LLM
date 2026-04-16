using System.Text.Json;
using System.Text.Json.Nodes;
using WebExpress.LLM.Model;

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
}
