using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace WebExpress.LLM.Tokenization;

/// <summary>
/// Implements a vocabulary-based tokenizer that maps tokens to IDs using a predefined vocabulary.
/// This serves as a foundation for more advanced tokenization schemes like SentencePiece.
/// </summary>
public sealed class VocabularyTokenizer : ITokenizer
{
    private readonly Dictionary<string, int> _tokenToId;
    private readonly Dictionary<int, string> _idToToken;
    private readonly int _unknownTokenId;

    /// <summary>
    /// Initializes a new instance of the VocabularyTokenizer class with the specified vocabulary
    /// and an optional ID for unknown tokens.
    /// </summary>
    /// <param name="vocabulary">
    /// The read‑only dictionary that assigns a unique ID to each token.  
    /// The dictionary must not be empty.
    /// </param>
    /// <param name="unknownTokenId">
    /// The ID used for unknown tokens. The default value is 0.
    /// </param>
    /// <exception cref="ArgumentException">
    /// Thrown when the provided vocabulary is empty.
    /// </exception>
    public VocabularyTokenizer(IReadOnlyDictionary<string, int> vocabulary, int unknownTokenId = 0)
    {
        ArgumentNullException.ThrowIfNull(vocabulary);

        if (vocabulary.Count == 0)
        {
            throw new ArgumentException("Vocabulary must not be empty.", nameof(vocabulary));
        }

        _tokenToId = new Dictionary<string, int>(vocabulary);
        _idToToken = vocabulary.ToDictionary(kvp => kvp.Value, kvp => kvp.Key);
        _unknownTokenId = unknownTokenId;
    }

    /// <summary>
    /// Encodes the specified text into a sequence of integer token identifiers.
    /// </summary>
    /// <remarks>Unknown tokens in the input text are mapped to a special identifier representing unknown
    /// tokens.</remarks>
    /// <param name="text">
    /// The text to encode. Cannot be null. If empty or consists only of white-space characters, an empty list is
    /// returned.
    /// </param>
    /// <returns>
    /// A read-only list of integer token identifiers representing the encoded text. Returns an empty list if the input
    /// text is null, empty, or consists only of white-space characters.
    /// </returns>
    public IReadOnlyList<int> Encode(string text)
    {
        ArgumentNullException.ThrowIfNull(text);

        if (string.IsNullOrWhiteSpace(text))
        {
            return Array.Empty<int>();
        }

        var tokens = SimpleTokenize(text);
        return tokens.Select(token => _tokenToId.TryGetValue(token, out var id) ? id : _unknownTokenId).ToArray();
    }

    /// <summary>
    /// Decodes a sequence of token IDs into the corresponding text representation.
    /// </summary>
    /// <remarks>
    /// If a token ID does not exist in the vocabulary, it is replaced with the string "<unk>" in the
    /// output.
    /// </remarks>
    /// <param name="tokens">
    /// The sequence of integer token IDs to decode. Cannot be null.
    /// </param>
    /// <returns>
    /// A string containing the decoded text. Unknown token IDs are represented as "<unk>".
    /// </returns>
    public string Decode(IEnumerable<int> tokens)
    {
        ArgumentNullException.ThrowIfNull(tokens);

        var textTokens = tokens
            .Select(id => _idToToken.TryGetValue(id, out var token) ? token : "<unk>")
            .ToArray();

        return string.Join("", textTokens);
    }

    /// <summary>
    /// Splits the specified text into tokens, treating each word and each whitespace character
    /// as a separate token.
    /// </summary>
    /// <remarks>
    /// Empty tokens are not returned. Whitespace characters are treated as individual tokens
    /// and are returned in the order in which they appear in the text.
    /// </remarks>
    /// <param name="text">
    /// The text to tokenize. May contain any sequence of characters.
    /// </param>
    /// <returns>
    /// An array of strings containing the extracted tokens.  
    /// Words and whitespace characters each appear as separate elements in the array.
    /// </returns>
    private static string[] SimpleTokenize(string text)
    {
        var tokens = new List<string>();
        var currentToken = new StringBuilder();

        foreach (var character in text)
        {
            if (char.IsWhiteSpace(character))
            {
                if (currentToken.Length > 0)
                {
                    tokens.Add(currentToken.ToString());
                    currentToken.Clear();
                }
                tokens.Add(character.ToString());
            }
            else
            {
                currentToken.Append(character);
            }
        }

        if (currentToken.Length > 0)
        {
            tokens.Add(currentToken.ToString());
        }

        return [.. tokens];
    }
}
