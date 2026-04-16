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

    public string Decode(IEnumerable<int> tokens)
    {
        ArgumentNullException.ThrowIfNull(tokens);

        var textTokens = tokens
            .Select(id => _idToToken.TryGetValue(id, out var token) ? token : "<unk>")
            .ToArray();

        return string.Join("", textTokens);
    }

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
