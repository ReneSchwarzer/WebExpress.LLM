using System.Text.Json;
using System.Text.Json.Nodes;
using WebExpress.LLM.Model;

namespace WebExpress.LLM.Test.Model;

public sealed class ModelConfigurationEosTokenIdTests
{
    [Fact]
    public void EosTokenId_ShouldDeserializeSingleInteger()
    {
        var json = """{"eos_token_id": 1}""";

        var config = JsonSerializer.Deserialize<ModelConfiguration>(json);

        Assert.NotNull(config);
        Assert.Equal(1, config.EosTokenId);
    }

    [Fact]
    public void EosTokenId_ShouldDeserializeArrayAndReturnFirstElement()
    {
        var json = """{"eos_token_id": [1, 106]}""";

        var config = JsonSerializer.Deserialize<ModelConfiguration>(json);

        Assert.NotNull(config);
        Assert.Equal(1, config.EosTokenId);
    }

    [Fact]
    public void EosTokenId_ShouldDeserializeArrayWithSingleElement()
    {
        var json = """{"eos_token_id": [42]}""";

        var config = JsonSerializer.Deserialize<ModelConfiguration>(json);

        Assert.NotNull(config);
        Assert.Equal(42, config.EosTokenId);
    }

    [Fact]
    public void TextConfig_EosTokenId_ShouldDeserializeSingleInteger()
    {
        var json = """{"eos_token_id": 2}""";

        var config = JsonSerializer.Deserialize<TextConfig>(json);

        Assert.NotNull(config);
        Assert.Equal(2, config.EosTokenId);
    }

    [Fact]
    public void TextConfig_EosTokenId_ShouldDeserializeArrayAndReturnFirstElement()
    {
        var json = """{"eos_token_id": [1, 106]}""";

        var config = JsonSerializer.Deserialize<TextConfig>(json);

        Assert.NotNull(config);
        Assert.Equal(1, config.EosTokenId);
    }

    [Fact]
    public void Load_ShouldSucceedWhenEosTokenIdIsArray()
    {
        var tempPath = Path.Combine(Path.GetTempPath(), Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(tempPath);

        try
        {
            var configJson = new JsonObject
            {
                ["model_type"] = "gemma",
                ["vocab_size"] = 256000,
                ["max_position_embeddings"] = 8192,
                ["hidden_size"] = 2048,
                ["intermediate_size"] = 8192,
                ["num_hidden_layers"] = 18,
                ["num_attention_heads"] = 8,
                ["num_key_value_heads"] = 1,
                ["head_dim"] = 256,
                ["eos_token_id"] = new JsonArray(1, 106)
            };

            File.WriteAllText(
                Path.Combine(tempPath, ModelLoader.DefaultConfigurationFileName),
                configJson.ToJsonString());
            File.WriteAllBytes(Path.Combine(tempPath, ModelLoader.DefaultWeightsFileName), [1, 2, 3, 4]);

            var loader = new ModelLoader();
            var model = loader.Load(tempPath);

            Assert.Equal(1, model.Configuration.EosTokenId);

            model.Dispose();
        }
        finally
        {
            Directory.Delete(tempPath, recursive: true);
        }
    }
}
