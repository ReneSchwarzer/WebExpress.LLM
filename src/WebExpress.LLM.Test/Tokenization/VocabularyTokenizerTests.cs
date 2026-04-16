using WebExpress.LLM.Tokenization;

namespace WebExpress.LLM.Test.Tokenization;

public sealed class VocabularyTokenizerTests
{
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

    [Fact]
    public void Constructor_WithEmptyVocabulary_ShouldThrowArgumentException()
    {
        var vocabulary = new Dictionary<string, int>();

        Assert.Throws<ArgumentException>(() => new VocabularyTokenizer(vocabulary));
    }

    [Fact]
    public void Encode_WithEmptyString_ShouldReturnEmpty()
    {
        var vocabulary = new Dictionary<string, int> { ["test"] = 1 };
        var tokenizer = new VocabularyTokenizer(vocabulary);

        var result = tokenizer.Encode("");

        Assert.Empty(result);
    }
}
