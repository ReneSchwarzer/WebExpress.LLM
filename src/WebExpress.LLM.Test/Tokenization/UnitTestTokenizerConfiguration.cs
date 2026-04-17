using WebExpress.LLM.Tokenization;

namespace WebExpress.LLM.Test.Tokenization;

/// <summary>
/// Provides a collection of unit tests for deserializing and validating tokenizer
/// configuration data.
/// </summary>
public sealed class UnitTestTokenizerConfiguration
{
    /// <summary>
    /// Verifies that the FromJson method correctly deserializes all fields from a valid JSON input into a
    /// TokenizerConfiguration instance.
    /// </summary>
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

    /// <summary>
    /// Verifies that the FromJson method of TokenizerConfiguration correctly assigns default values to properties when
    /// provided with minimal JSON input.
    /// </summary>
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

    /// <summary>
    /// Verifies that the <c>FromJson</c> method throws an <see cref="ArgumentException"/>
    /// when an empty string is provided.
    /// </summary>
    [Fact]
    public void FromJson_WithEmptyString_ShouldThrowArgumentException()
    {
        Assert.Throws<ArgumentException>(() => TokenizerConfiguration.FromJson(""));
    }

    /// <summary>
    /// Verifies that the <c>FromJson</c> method throws an <see cref="ArgumentException"/>
    /// when the input string is null.
    /// </summary>
    [Fact]
    public void FromJson_WithNullString_ShouldThrowArgumentException()
    {
        Assert.Throws<ArgumentException>(() => TokenizerConfiguration.FromJson(null));
    }

    /// <summary>
    /// Verifies that the <c>FromFile</c> method throws a <see cref="FileNotFoundException"/>
    /// when a non‑existent file is specified.
    /// </summary>
    [Fact]
    public void FromFile_WithNonexistentFile_ShouldThrowFileNotFoundException()
    {
        Assert.Throws<FileNotFoundException>(() =>
            TokenizerConfiguration.FromFile("/nonexistent/path/tokenizer_config.json"));
    }

    /// <summary>
    /// Verifies that the <c>FromFile</c> method throws an <see cref="ArgumentException"/>
    /// when an empty path is provided.
    /// </summary>
    [Fact]
    public void FromFile_WithEmptyPath_ShouldThrowArgumentException()
    {
        Assert.Throws<ArgumentException>(() => TokenizerConfiguration.FromFile(""));
    }

    /// <summary>
    /// Verifies that the FromFile method correctly loads a TokenizerConfiguration from a valid configuration file.
    /// </summary>
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

    /// <summary>
    /// Verifies that the FromJson method clamps extremely large values for model_max_length to long.MaxValue when
    /// deserializing a TokenizerConfiguration from JSON.
    /// </summary>
    [Fact]
    public void FromJson_WithLargeFloatModelMaxLength_ShouldClampToLongMaxValue()
    {
        // HuggingFace tokenizer_config.json files often set model_max_length to a huge float
        // (1e30 / 1000000000000000019884624838656) to indicate "unlimited context".
        var json = """{ "model_max_length": 1000000000000000019884624838656 }""";

        var config = TokenizerConfiguration.FromJson(json);

        Assert.Equal(long.MaxValue, config.ModelMaxLength);
    }

    /// <summary>
    /// Verifies that the FromJson method clamps the ModelMaxLength property to long.MaxValue when the JSON input
    /// specifies a value exceeding the maximum value for a long integer.
    /// </summary>
    [Fact]
    public void FromJson_WithExponentialModelMaxLength_ShouldClampToLongMaxValue()
    {
        var json = """{ "model_max_length": 1e30 }""";

        var config = TokenizerConfiguration.FromJson(json);

        Assert.Equal(long.MaxValue, config.ModelMaxLength);
    }
}
