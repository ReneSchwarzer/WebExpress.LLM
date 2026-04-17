using System;
using System.Collections.Generic;
using System.IO;
using System.Text;
using System.Text.Json;
using System.Text.RegularExpressions;

namespace WebExpress.LLM.Tokenization;

/// <summary>
/// A dedicated tokenizer for Google Gemma models that reads and interprets both
/// <c>tokenizer.json</c> (HuggingFace tokenizers format) and <c>tokenizer_config.json</c>.
/// </summary>
/// <remarks>
/// This tokenizer provides full support for the Gemma tokenization pipeline including:
/// <list type="bullet">
///   <item>Model loading from <c>tokenizer.json</c> (vocabulary and BPE merge rules)</item>
///   <item>Configuration loading from <c>tokenizer_config.json</c> (special tokens, flags)</item>
///   <item>NFKC normalization of input text</item>
///   <item>Pre-tokenization using whitespace and punctuation splitting</item>
///   <item>BPE-based subword encoding using iterative pair merging</item>
///   <item>Special token handling (BOS, EOS, UNK, PAD) with configurable token names</item>
///   <item>Decoding with SentencePiece whitespace convention (▁ prefix)</item>
/// </list>
/// </remarks>
public sealed class GemmaTokenizer : ITokenizer
{
    /// <summary>
    /// The Unicode character used by SentencePiece/Gemma to represent a space/word boundary.
    /// </summary>
    internal const char SpaceSymbol = '\u2581';

    private readonly Dictionary<string, int> _pieceToId;
    private readonly Dictionary<int, string> _idToPiece;
    private readonly Dictionary<(string Left, string Right), int> _mergeRank;
    private readonly int _unknownTokenId;
    private readonly int _bosTokenId;
    private readonly int _eosTokenId;
    private readonly bool _addBosToken;
    private readonly bool _addEosToken;

    /// <summary>
    /// Initializes a new instance of the <see cref="GemmaTokenizer"/> class with the specified
    /// vocabulary, merge rules, and token configuration.
    /// </summary>
    /// <param name="vocabulary">The mapping from piece strings to token IDs.</param>
    /// <param name="merges">
    /// Ordered list of BPE merge rules. Each entry is a pair of pieces that should be merged,
    /// with earlier entries having higher priority.
    /// </param>
    /// <param name="unknownTokenId">The ID for unknown tokens. Default is 0.</param>
    /// <param name="bosTokenId">The ID for the beginning-of-sequence token. Default is 2.</param>
    /// <param name="eosTokenId">The ID for the end-of-sequence token. Default is 1.</param>
    /// <param name="addBosToken">Whether to prepend a BOS token during encoding. Default is true.</param>
    /// <param name="addEosToken">Whether to append an EOS token during encoding. Default is false.</param>
    public GemmaTokenizer(
        IReadOnlyDictionary<string, int> vocabulary,
        IReadOnlyList<(string Left, string Right)> merges,
        int unknownTokenId = 0,
        int bosTokenId = 2,
        int eosTokenId = 1,
        bool addBosToken = true,
        bool addEosToken = false)
    {
        ArgumentNullException.ThrowIfNull(vocabulary);
        ArgumentNullException.ThrowIfNull(merges);

        if (vocabulary.Count == 0)
        {
            throw new ArgumentException("Vocabulary must not be empty.", nameof(vocabulary));
        }

        _pieceToId = new Dictionary<string, int>(vocabulary);
        _idToPiece = new Dictionary<int, string>(vocabulary.Count);

        foreach (var kvp in vocabulary)
        {
            _idToPiece[kvp.Value] = kvp.Key;
        }

        _mergeRank = new Dictionary<(string, string), int>(merges.Count);

        for (var i = 0; i < merges.Count; i++)
        {
            _mergeRank[merges[i]] = i;
        }

        _unknownTokenId = unknownTokenId;
        _bosTokenId = bosTokenId;
        _eosTokenId = eosTokenId;
        _addBosToken = addBosToken;
        _addEosToken = addEosToken;
    }

    /// <summary>
    /// Encodes the specified text into a sequence of integer token identifiers using BPE.
    /// The input is first normalized (NFKC), then pre-tokenized, then each word is encoded
    /// using BPE merge rules.
    /// </summary>
    /// <param name="text">The text to encode. Cannot be null.</param>
    /// <returns>
    /// A read-only list of integers representing the encoded tokens. Includes optional BOS/EOS tokens
    /// according to the tokenizer configuration.
    /// </returns>
    public IReadOnlyList<int> Encode(string text)
    {
        ArgumentNullException.ThrowIfNull(text);

        if (string.IsNullOrEmpty(text))
        {
            var empty = new List<int>();

            if (_addBosToken)
            {
                empty.Add(_bosTokenId);
            }

            if (_addEosToken)
            {
                empty.Add(_eosTokenId);
            }

            return empty;
        }

        // Step 1: Normalize (NFKC)
        var normalized = Normalize(text);

        // Step 2: Pre-tokenize (split into words)
        var words = PreTokenize(normalized);

        // Step 3: Encode via BPE
        var tokenIds = new List<int>();

        if (_addBosToken)
        {
            tokenIds.Add(_bosTokenId);
        }

        foreach (var word in words)
        {
            var wordTokens = ApplyBpe(word);

            foreach (var token in wordTokens)
            {
                tokenIds.Add(_pieceToId.TryGetValue(token, out var id) ? id : _unknownTokenId);
            }
        }

        if (_addEosToken)
        {
            tokenIds.Add(_eosTokenId);
        }

        return tokenIds;
    }

    /// <summary>
    /// Decodes a sequence of integer tokens into the corresponding string representation.
    /// Replaces the SentencePiece space symbol (▁) with actual spaces and strips the leading space.
    /// </summary>
    /// <param name="tokens">The sequence of integer tokens to decode. Cannot be null.</param>
    /// <returns>A string representing the decoded text.</returns>
    public string Decode(IEnumerable<int> tokens)
    {
        ArgumentNullException.ThrowIfNull(tokens);

        var sb = new StringBuilder();

        foreach (var tokenId in tokens)
        {
            // Skip special tokens in decode output
            if (tokenId == _bosTokenId || tokenId == _eosTokenId)
            {
                continue;
            }

            if (_idToPiece.TryGetValue(tokenId, out var piece))
            {
                sb.Append(piece);
            }
            else
            {
                sb.Append("<unk>");
            }
        }

        // Replace the SentencePiece space symbol with actual spaces
        sb.Replace(SpaceSymbol, ' ');

        // Remove leading space that results from the initial ▁ prefix
        if (sb.Length > 0 && sb[0] == ' ')
        {
            sb.Remove(0, 1);
        }

        return sb.ToString();
    }

    /// <summary>
    /// Creates a <see cref="GemmaTokenizer"/> from a HuggingFace <c>tokenizer.json</c> file
    /// and an optional <c>tokenizer_config.json</c> configuration.
    /// </summary>
    /// <param name="tokenizerJsonPath">Path to the <c>tokenizer.json</c> file.</param>
    /// <param name="config">
    /// Optional tokenizer configuration loaded from <c>tokenizer_config.json</c>.
    /// When provided, its <c>BosToken</c>/<c>EosToken</c>/<c>UnkToken</c> strings are used to look
    /// up the correct token IDs from the vocabulary.
    /// </param>
    /// <returns>A new <see cref="GemmaTokenizer"/> instance.</returns>
    /// <exception cref="ArgumentException">Thrown when <paramref name="tokenizerJsonPath"/> is null or whitespace.</exception>
    /// <exception cref="FileNotFoundException">Thrown when the file does not exist.</exception>
    /// <exception cref="InvalidDataException">Thrown when the file lacks a valid <c>model</c> section or vocabulary.</exception>
    public static GemmaTokenizer FromTokenizerJson(
        string tokenizerJsonPath,
        TokenizerConfiguration config = null)
    {
        if (string.IsNullOrWhiteSpace(tokenizerJsonPath))
        {
            throw new ArgumentException("Tokenizer JSON path must be provided.", nameof(tokenizerJsonPath));
        }

        if (!File.Exists(tokenizerJsonPath))
        {
            throw new FileNotFoundException("tokenizer.json was not found.", tokenizerJsonPath);
        }

        var json = File.ReadAllText(tokenizerJsonPath);
        using var doc = JsonDocument.Parse(json);
        var root = doc.RootElement;

        if (!root.TryGetProperty("model", out var modelElement))
        {
            throw new InvalidDataException("tokenizer.json does not contain a 'model' section.");
        }

        // Read vocabulary
        var vocab = new Dictionary<string, int>();

        if (modelElement.TryGetProperty("vocab", out var vocabElement))
        {
            foreach (var kv in vocabElement.EnumerateObject())
            {
                vocab[kv.Name] = kv.Value.GetInt32();
            }
        }

        if (vocab.Count == 0)
        {
            throw new InvalidDataException("tokenizer.json contains no vocabulary entries.");
        }

        // Read merge rules.
        // HuggingFace tokenizer.json uses two formats:
        //   • String format: "a b"  (e.g. GPT-2, LLaMA)
        //   • Array format:  ["a", "b"]  (e.g. Gemma)
        // Both are handled here.
        var mergePairs = new List<(string, string)>();

        if (modelElement.TryGetProperty("merges", out var mergesElement))
        {
            foreach (var m in mergesElement.EnumerateArray())
            {
                if (m.ValueKind == JsonValueKind.Array)
                {
                    // Array format: ["left", "right"]
                    var enumerator = m.EnumerateArray();

                    if (!enumerator.MoveNext()) { continue; }
                    var left = enumerator.Current.GetString();

                    if (!enumerator.MoveNext()) { continue; }
                    var right = enumerator.Current.GetString();

                    if (!string.IsNullOrEmpty(left) && !string.IsNullOrEmpty(right))
                    {
                        mergePairs.Add((left, right));
                    }
                }
                else if (m.ValueKind == JsonValueKind.String)
                {
                    // String format: "left right"
                    var s = m.GetString();

                    if (string.IsNullOrEmpty(s))
                    {
                        continue;
                    }

                    var parts = s.Split(' ', 2, StringSplitOptions.RemoveEmptyEntries);

                    if (parts.Length == 2)
                    {
                        mergePairs.Add((parts[0], parts[1]));
                    }
                }
            }
        }

        // Resolve special token IDs from vocabulary.
        // Gemma uses <bos>/<eos> rather than <s>/</s>.
        var unkId = ResolveTokenId(vocab, config?.UnkToken, "<unk>", 0);
        var bosId = ResolveTokenId(vocab, config?.BosToken, "<bos>", 2);
        var eosId = ResolveTokenId(vocab, config?.EosToken, "<eos>", 1);

        var addBos = config?.AddBosToken ?? true;
        var addEos = config?.AddEosToken ?? false;

        return new GemmaTokenizer(vocab, mergePairs, unkId, bosId, eosId, addBos, addEos);
    }

    /// <summary>
    /// Creates a <see cref="GemmaTokenizer"/> from explicit vocabulary, merge list, and
    /// optional configuration. Useful for testing or when data comes from another source.
    /// </summary>
    /// <param name="vocabulary">Mapping of piece strings to token IDs.</param>
    /// <param name="merges">Ordered list of BPE merge rules as space-separated strings (e.g. "a b").</param>
    /// <param name="config">Optional tokenizer configuration for special token settings.</param>
    /// <returns>A new <see cref="GemmaTokenizer"/> instance.</returns>
    public static GemmaTokenizer FromVocabularyAndMerges(
        IReadOnlyDictionary<string, int> vocabulary,
        IReadOnlyList<string> merges,
        TokenizerConfiguration config = null)
    {
        ArgumentNullException.ThrowIfNull(vocabulary);
        ArgumentNullException.ThrowIfNull(merges);

        var mergePairs = new List<(string, string)>(merges.Count);

        foreach (var merge in merges)
        {
            var parts = merge.Split(' ', 2, StringSplitOptions.RemoveEmptyEntries);

            if (parts.Length == 2)
            {
                mergePairs.Add((parts[0], parts[1]));
            }
        }

        var unkId = ResolveTokenId(vocabulary, config?.UnkToken, "<unk>", 0);
        var bosId = ResolveTokenId(vocabulary, config?.BosToken, "<bos>", 2);
        var eosId = ResolveTokenId(vocabulary, config?.EosToken, "<eos>", 1);

        var addBos = config?.AddBosToken ?? true;
        var addEos = config?.AddEosToken ?? false;

        return new GemmaTokenizer(vocabulary, mergePairs, unkId, bosId, eosId, addBos, addEos);
    }

    /// <summary>
    /// Applies NFKC normalization to the input text.
    /// This is the standard normalization used by SentencePiece and Gemma tokenizers.
    /// </summary>
    /// <param name="text">The text to normalize.</param>
    /// <returns>The NFKC-normalized text.</returns>
    internal static string Normalize(string text)
    {
        return text.Normalize(NormalizationForm.FormKC);
    }

    /// <summary>
    /// Splits input text into words following SentencePiece conventions.
    /// Every word is prefixed with the ▁ (U+2581) character to represent a word boundary,
    /// including the very first word (matching standard SentencePiece behavior).
    /// </summary>
    internal static List<string> PreTokenize(string text)
    {
        var words = new List<string>();
        var current = new StringBuilder();

        for (var i = 0; i < text.Length; i++)
        {
            var ch = text[i];

            if (char.IsWhiteSpace(ch))
            {
                if (current.Length > 0)
                {
                    words.Add(current.ToString());
                    current.Clear();
                }
            }
            else
            {
                if (current.Length == 0)
                {
                    // Every word starts with ▁ (including the first word)
                    current.Append(SpaceSymbol);
                }

                current.Append(ch);
            }
        }

        if (current.Length > 0)
        {
            words.Add(current.ToString());
        }

        return words;
    }

    /// <summary>
    /// Resolves a special token ID from the vocabulary, preferring <paramref name="configTokenName"/>
    /// (from tokenizer config) over <paramref name="fallbackTokenName"/> (Gemma default like "&lt;bos&gt;"),
    /// and finally returning <paramref name="defaultId"/> when neither is present in the vocabulary.
    /// </summary>
    private static int ResolveTokenId(
        IReadOnlyDictionary<string, int> vocab,
        string configTokenName,
        string fallbackTokenName,
        int defaultId)
    {
        if (!string.IsNullOrEmpty(configTokenName) && vocab.TryGetValue(configTokenName, out var configId))
        {
            return configId;
        }

        return vocab.TryGetValue(fallbackTokenName, out var fallbackId) ? fallbackId : defaultId;
    }

    /// <summary>
    /// Applies BPE merge operations to a single word, producing a list of subword tokens.
    /// </summary>
    private List<string> ApplyBpe(string word)
    {
        if (word.Length == 0)
        {
            return [];
        }

        // Check if the entire word is in vocabulary
        if (_pieceToId.ContainsKey(word))
        {
            return [word];
        }

        // Start with individual characters as the initial token sequence
        var symbols = new List<string>(word.Length);

        foreach (var ch in word)
        {
            symbols.Add(ch.ToString());
        }

        while (symbols.Count > 1)
        {
            // Find the highest-priority merge pair (lowest rank)
            var bestRank = int.MaxValue;
            var bestIndex = -1;

            for (var i = 0; i < symbols.Count - 1; i++)
            {
                var pair = (symbols[i], symbols[i + 1]);

                if (_mergeRank.TryGetValue(pair, out var rank) && rank < bestRank)
                {
                    bestRank = rank;
                    bestIndex = i;
                }
            }

            if (bestIndex < 0)
            {
                break; // No more merges possible
            }

            // Apply the merge
            var merged = symbols[bestIndex] + symbols[bestIndex + 1];
            symbols[bestIndex] = merged;
            symbols.RemoveAt(bestIndex + 1);
        }

        return symbols;
    }
}
