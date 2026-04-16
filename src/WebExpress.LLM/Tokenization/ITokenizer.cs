using System.Collections.Generic;

namespace WebExpress.LLM.Tokenization;

/// <summary>
/// Defines methods for encoding text into token sequences and decoding token sequences back into text.
/// </summary>
public interface ITokenizer
{
    /// <summary>
    /// Encodes the specified text into a sequence of integer token identifiers.
    /// </summary>
    /// <param name="text">The text to encode. Cannot be null.</param>
    /// <returns>
    /// A read-only list of integers representing the encoded tokens of the input text. The list will be 
    /// empty if the input text is empty.
    /// </returns>
    IReadOnlyList<int> Encode(string text);

    /// <summary>
    /// Decodes a sequence of integer tokens into the corresponding string representation.
    /// </summary>
    /// <param name="tokens">The sequence of integer tokens to decode. Cannot be null.</param>
    /// <returns>A string representing the decoded value of the input tokens.</returns>
    string Decode(IEnumerable<int> tokens);
}
