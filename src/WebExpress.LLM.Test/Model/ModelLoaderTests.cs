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
                ContextLength = 8192
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
            Assert.Equal([1, 2, 3, 4], model.Weights);
        }
        finally
        {
            Directory.Delete(tempPath, recursive: true);
        }
    }
}
