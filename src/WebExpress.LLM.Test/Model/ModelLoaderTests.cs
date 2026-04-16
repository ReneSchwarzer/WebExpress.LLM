using System.Text.Json;
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
