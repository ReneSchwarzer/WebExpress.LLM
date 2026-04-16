using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace WebExpress.LLM.Tokenization;

/// <summary>
/// Provides methods for encoding text into a sequence of byte tokens and decoding byte tokens back into text using
/// UTF-8 encoding.
/// </summary>
public sealed class ByteTokenizer : ITokenizer
{
    /// <summary>
    /// Encodes the specified string as a read‑only list of integers, where each element represents  
    /// the UTF‑8 byte of a character position.
    /// </summary>
    /// <param name="text">The string to encode. Must not be null.</param>
    /// <returns>A read‑only list of integers containing the UTF‑8 encoded bytes of the input string.</returns>
    /// <exception cref="ArgumentNullException">Thrown when <paramref name="text"/> is null.</exception>
    public IReadOnlyList<int> Encode(string text)
    {
        return text is null
            ? throw new ArgumentNullException(nameof(text))
            : (IReadOnlyList<int>)Encoding.UTF8.GetBytes(text).Select(static value => (int)value).ToArray();
    }

    /// <summary>
    /// Decodes a sequence of integer tokens representing UTF-8 encoded bytes into a string.
    /// </summary>
    /// <param name="tokens">
    /// The sequence of integer tokens to decode. Each token must be within the range of 0 to 255, inclusive.
    /// </param>
    /// <returns>
    /// A string decoded from the specified UTF-8 byte tokens.
    /// </returns>
    /// <exception cref="ArgumentNullException">
    /// Thrown if <paramref name="tokens"/> is <see langword="null"/>.
    /// </exception>
    /// <exception cref="ArgumentOutOfRangeException">
    /// Thrown if any token in <paramref name="tokens"/> is less than 0 or greater than 255.
    /// </exception>
    public string Decode(IEnumerable<int> tokens)
    {
        ArgumentNullException.ThrowIfNull(tokens);

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
