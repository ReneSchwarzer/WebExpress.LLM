using WebExpress.LLM.Tokenization;

namespace WebExpress.LLM.Test.Tokenization;

/// <summary>
/// Provides unit tests for the ByteTokenizer, ensuring correct byte-level encoding and decoding.
/// </summary>
public sealed class UnitTestByteTokenizer
{
    /// <summary>
    /// Tests that the encoding and decoding process is deterministic and reversible.
    /// </summary>
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
