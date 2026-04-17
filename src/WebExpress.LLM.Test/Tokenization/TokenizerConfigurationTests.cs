using WebExpress.LLM.Tokenization;

namespace WebExpress.LLM.Test.Tokenization;

public sealed class TokenizerConfigurationTests
{
    [Fact]
    public void FromJson_WithValidJson_ShouldDeserializeAllFields()
    {
        var json = """
        {
            "backend": "tokenizers",
            "tokenizer_class": "GemmaTokenizer",
            "bos_token": "<s>",
            "eos_token": "</s>",
            "unk_token": "<unk>",
            "pad_token": "<pad>",
            "audio_token": "<|audio|>",
            "boa_token": "<|begin_audio|>",
            "boi_token": "<|begin_image|>",
            "model_max_length": 32768,
            "add_bos_token": true,
            "add_eos_token": false,
            "clean_up_tokenization_spaces": false,
            "additional_special_tokens": ["<|special1|>", "<|special2|>"]
        }
        """;

        var config = TokenizerConfiguration.FromJson(json);

        Assert.Equal("tokenizers", config.Backend);
        Assert.Equal("GemmaTokenizer", config.TokenizerClass);
        Assert.Equal("<s>", config.BosToken);
        Assert.Equal("</s>", config.EosToken);
        Assert.Equal("<unk>", config.UnkToken);
        Assert.Equal("<pad>", config.PadToken);
        Assert.Equal("<|audio|>", config.AudioToken);
        Assert.Equal("<|begin_audio|>", config.BoaToken);
        Assert.Equal("<|begin_image|>", config.BoiToken);
        Assert.Equal(32768, config.ModelMaxLength);
        Assert.True(config.AddBosToken);
        Assert.False(config.AddEosToken);
        Assert.False(config.CleanUpTokenizationSpaces);
        Assert.Equal(2, config.AdditionalSpecialTokens.Count);
        Assert.Equal("<|special1|>", config.AdditionalSpecialTokens[0]);
    }

    [Fact]
    public void FromJson_WithMinimalJson_ShouldUseDefaults()
    {
        var json = """{ "backend": "tokenizers" }""";

        var config = TokenizerConfiguration.FromJson(json);

        Assert.Equal("tokenizers", config.Backend);
        Assert.Equal(string.Empty, config.BosToken);
        Assert.Equal(string.Empty, config.EosToken);
        Assert.False(config.AddBosToken);
        Assert.False(config.AddEosToken);
        Assert.Equal(0, config.ModelMaxLength);
        Assert.Empty(config.AdditionalSpecialTokens);
    }

    [Fact]
    public void FromJson_WithEmptyString_ShouldThrowArgumentException()
    {
        Assert.Throws<ArgumentException>(() => TokenizerConfiguration.FromJson(""));
    }

    [Fact]
    public void FromJson_WithNullString_ShouldThrowArgumentException()
    {
        Assert.Throws<ArgumentException>(() => TokenizerConfiguration.FromJson(null));
    }

    [Fact]
    public void FromFile_WithNonexistentFile_ShouldThrowFileNotFoundException()
    {
        Assert.Throws<FileNotFoundException>(() =>
            TokenizerConfiguration.FromFile("/nonexistent/path/tokenizer_config.json"));
    }

    [Fact]
    public void FromFile_WithEmptyPath_ShouldThrowArgumentException()
    {
        Assert.Throws<ArgumentException>(() => TokenizerConfiguration.FromFile(""));
    }

    [Fact]
    public void FromFile_WithValidFile_ShouldLoadConfiguration()
    {
        var tempFile = Path.GetTempFileName();

        try
        {
            var json = """
            {
                "backend": "tokenizers",
                "bos_token": "<s>",
                "eos_token": "</s>",
                "model_max_length": 8192
            }
            """;

            File.WriteAllText(tempFile, json);

            var config = TokenizerConfiguration.FromFile(tempFile);

            Assert.Equal("tokenizers", config.Backend);
            Assert.Equal("<s>", config.BosToken);
            Assert.Equal("</s>", config.EosToken);
            Assert.Equal(8192, config.ModelMaxLength);
        }
        finally
        {
            File.Delete(tempFile);
        }
    }
}
