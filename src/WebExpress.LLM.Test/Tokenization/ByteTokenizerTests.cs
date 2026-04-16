using WebExpress.LLM.Tokenization;

namespace WebExpress.LLM.Test.Tokenization;

public sealed class ByteTokenizerTests
{
    [Fact]
    public void EncodeDecode_ShouldBeDeterministicAndReversible()
    {
        var tokenizer = new ByteTokenizer();
        const string text = "Hello Gemma 4 👋";

        var firstEncoding = tokenizer.Encode(text);
        var secondEncoding = tokenizer.Encode(text);

        Assert.Equal(firstEncoding, secondEncoding);
        Assert.Equal(text, tokenizer.Decode(firstEncoding));
    }
}
