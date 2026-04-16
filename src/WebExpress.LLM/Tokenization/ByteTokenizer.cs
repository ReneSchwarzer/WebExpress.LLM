using System.Text;

namespace WebExpress.LLM.Tokenization;

public sealed class ByteTokenizer : ITokenizer
{
    public IReadOnlyList<int> Encode(string text)
    {
        if (text is null)
        {
            throw new ArgumentNullException(nameof(text));
        }

        return Encoding.UTF8.GetBytes(text).Select(static value => (int)value).ToArray();
    }

    public string Decode(IEnumerable<int> tokens)
    {
        if (tokens is null)
        {
            throw new ArgumentNullException(nameof(tokens));
        }

        var bytes = tokens.Select(static token =>
        {
            if (token is < byte.MinValue or > byte.MaxValue)
            {
                throw new ArgumentOutOfRangeException(nameof(tokens), token, "Token values must be in byte range.");
            }

            return (byte)token;
        }).ToArray();

        return Encoding.UTF8.GetString(bytes);
    }
}
