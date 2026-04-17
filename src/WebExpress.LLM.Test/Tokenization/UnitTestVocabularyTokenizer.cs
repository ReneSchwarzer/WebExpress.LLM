using WebExpress.LLM.Tokenization;

namespace WebExpress.LLM.Test.Tokenization;

/// <summary>
/// Provides a collection of unit tests for the <see cref="VocabularyTokenizer"/> class,
/// validating its encoding and decoding functionality as well as its error‑handling behavior.
/// </summary>
public sealed class UnitTestVocabularyTokenizer
{
    /// <summary>
    /// Verifies that encoding and subsequently decoding a piece of text using a simple vocabulary
    /// mapping is reversible.
    /// </summary>
    [Fact]
    public void EncodeDecode_WithSimpleVocabulary_ShouldBeReversible()
    {
        var vocabulary = new Dictionary<string, int>
        {
            ["<unk>"] = 0,
            ["Hello"] = 1,
            [" "] = 2,
            ["world"] = 3,
            ["!"] = 4
        };

        var tokenizer = new VocabularyTokenizer(vocabulary, unknownTokenId: 0);
        var text = "Hello world";

        var tokens = tokenizer.Encode(text);
        var decoded = tokenizer.Decode(tokens);

        Assert.Equal("Hello world", decoded);
    }

    /// <summary>
    /// Verifies that the Encode method assigns the unknown token ID when an input token is not present in the
    /// vocabulary.
    /// </summary>
    [Fact]
    public void Encode_WithUnknownToken_ShouldUseUnknownId()
    {
        var vocabulary = new Dictionary<string, int>
        {
            ["<unk>"] = 0,
            ["Hello"] = 1
        };

        var tokenizer = new VocabularyTokenizer(vocabulary, unknownTokenId: 0);
        var tokens = tokenizer.Encode("Hello unknown");

        Assert.Contains(0, tokens);
        Assert.Contains(1, tokens);
    }

    /// <summary>
    /// Verifies that the constructor of <see cref="VocabularyTokenizer"/> throws an
    /// <see cref="ArgumentException"/> when an empty vocabulary is provided.
    /// </summary>
    [Fact]
    public void Constructor_WithEmptyVocabulary_ShouldThrowArgumentException()
    {
        var vocabulary = new Dictionary<string, int>();

        Assert.Throws<ArgumentException>(() => new VocabularyTokenizer(vocabulary));
    }

    /// <summary>
    /// Verifies that the Encode method returns an empty collection when provided with an empty string as input.
    /// </summary>
    [Fact]
    public void Encode_WithEmptyString_ShouldReturnEmpty()
    {
        var vocabulary = new Dictionary<string, int> { ["test"] = 1 };
        var tokenizer = new VocabularyTokenizer(vocabulary);

        var result = tokenizer.Encode("");

        Assert.Empty(result);
    }
}
