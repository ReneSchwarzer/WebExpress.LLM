using System;
using System.Collections.Generic;
using System.IO;
using System.Text;
using System.Text.Json;

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
///   <item>Normalization matching the tokenizer.json pipeline: NFKC followed by Replace(" ", "▁")</item>
///   <item>BPE-based subword encoding using iterative pair merging over the full normalized string</item>
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
    private readonly string[] _specialTokens;

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
    /// <param name="specialTokens">
    /// Optional list of special / added tokens that must be matched as whole atoms before
    /// BPE runs (e.g. <c>&lt;bos&gt;</c>, <c>&lt;|turn&gt;</c>). When null, the set is
    /// auto-derived from the vocabulary: any entry of length &gt; 1 that starts with <c>&lt;</c>
    /// and ends with <c>&gt;</c> is treated as a special token. Every special token must
    /// exist in <paramref name="vocabulary"/>.
    /// </param>
    public GemmaTokenizer(
        IReadOnlyDictionary<string, int> vocabulary,
        IReadOnlyList<(string Left, string Right)> merges,
        int unknownTokenId = 0,
        int bosTokenId = 2,
        int eosTokenId = 1,
        bool addBosToken = true,
        bool addEosToken = false,
        IReadOnlyCollection<string> specialTokens = null)
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

        // Pre-compute the special-token list, sorted by length (descending) so the
        // longest-match wins during the pre-split scan (e.g. "<start_of_turn>"
        // must win over "<start>" when both are present).
        var specials = new List<string>();

        if (specialTokens != null)
        {
            foreach (var s in specialTokens)
            {
                if (!string.IsNullOrEmpty(s) && _pieceToId.ContainsKey(s))
                {
                    specials.Add(s);
                }
            }
        }
        else
        {
            foreach (var key in _pieceToId.Keys)
            {
                if (key.Length > 1 && key[0] == '<' && key[^1] == '>')
                {
                    specials.Add(key);
                }
            }
        }

        specials.Sort((a, b) => b.Length.CompareTo(a.Length));
        _specialTokens = specials.ToArray();
    }

    /// <summary>
    /// Encodes the specified text into a sequence of integer token identifiers using BPE.
    /// The input is first scanned for special / added tokens (e.g. <c>&lt;bos&gt;</c>,
    /// <c>&lt;|turn&gt;</c>) which are emitted as atomic IDs. Any remaining runs of
    /// ordinary text are normalized (NFKC followed by " " → "▁", matching the
    /// tokenizer.json normalizer) and handed to BPE. The pre_tokenizer declared in
    /// tokenizer.json splits on " " after the normalizer has already consumed every
    /// regular space, so it is a no-op in practice and is omitted.
    /// </summary>
    /// <param name="text">The text to encode. Cannot be null.</param>
    /// <returns>
    /// A read-only list of integers representing the encoded tokens. Includes optional BOS/EOS tokens
    /// according to the tokenizer configuration.
    /// </returns>
    public IReadOnlyList<int> Encode(string text)
    {
        ArgumentNullException.ThrowIfNull(text);

        var tokenIds = new List<int>();

        if (_addBosToken)
        {
            tokenIds.Add(_bosTokenId);
        }

        if (!string.IsNullOrEmpty(text))
        {
            var buffer = new StringBuilder();
            var i = 0;

            while (i < text.Length)
            {
                var matched = MatchSpecialToken(text, i);

                if (matched != null)
                {
                    if (buffer.Length > 0)
                    {
                        EncodeSegment(buffer.ToString(), tokenIds);
                        buffer.Clear();
                    }

                    tokenIds.Add(_pieceToId[matched]);
                    i += matched.Length;
                }
                else
                {
                    buffer.Append(text[i]);
                    i++;
                }
            }

            if (buffer.Length > 0)
            {
                EncodeSegment(buffer.ToString(), tokenIds);
            }
        }

        if (_addEosToken)
        {
            tokenIds.Add(_eosTokenId);
        }

        return tokenIds;
    }

    /// <summary>
    /// Returns the longest special token that matches at <paramref name="start"/> in
    /// <paramref name="text"/>, or <c>null</c> if none does. The candidate list is
    /// pre-sorted by descending length, so the first hit is the longest.
    /// </summary>
    private string MatchSpecialToken(string text, int start)
    {
        foreach (var token in _specialTokens)
        {
            if (start + token.Length <= text.Length &&
                string.CompareOrdinal(text, start, token, 0, token.Length) == 0)
            {
                return token;
            }
        }

        return null;
    }

    /// <summary>
    /// Normalizes a non-special text segment and appends the BPE-encoded tokens to
    /// <paramref name="tokenIds"/>.
    /// </summary>
    private void EncodeSegment(string segment, List<int> tokenIds)
    {
        var normalized = Normalize(segment);
        var pieces = ApplyBpe(normalized);

        foreach (var piece in pieces)
        {
            tokenIds.Add(_pieceToId.TryGetValue(piece, out var id) ? id : _unknownTokenId);
        }
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

        // Read the optional added_tokens array. HuggingFace stores control /
        // special tokens here (e.g. <bos>, <eos>, <|turn>, <start_of_turn>)
        // together with their IDs; they must be matched atomically before BPE
        // runs, otherwise a substring like "<|turn>" would be shattered into
        // per-character BPE units.
        List<string> addedTokens = null;

        if (root.TryGetProperty("added_tokens", out var addedElement) &&
            addedElement.ValueKind == JsonValueKind.Array)
        {
            addedTokens = [];

            foreach (var entry in addedElement.EnumerateArray())
            {
                if (entry.TryGetProperty("content", out var contentElement) &&
                    contentElement.ValueKind == JsonValueKind.String)
                {
                    var content = contentElement.GetString();

                    if (!string.IsNullOrEmpty(content) && vocab.ContainsKey(content))
                    {
                        addedTokens.Add(content);
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

        return new GemmaTokenizer(vocab, mergePairs, unkId, bosId, eosId, addBos, addEos, addedTokens);
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
    /// Applies the normalization pipeline declared in Gemma's tokenizer.json: NFKC
    /// Unicode normalization followed by Replace(" ", "▁"). The vocabulary stores
    /// word-initial tokens with a leading ▁, so "Hallo" and " Hallo" must remain
    /// distinguishable after normalization.
    /// </summary>
    /// <param name="text">The text to normalize.</param>
    /// <returns>The normalized text with regular spaces replaced by ▁.</returns>
    internal static string Normalize(string text)
    {
        return text.Normalize(NormalizationForm.FormKC).Replace(' ', SpaceSymbol);
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
    /// The initial split uses a greedy longest-prefix match against the full vocabulary
    /// (not character-by-character). This matches how Gemma's reference tokenizer resolves
    /// multi-character atomic vocab entries such as <c>Hallo</c>, <c>▁Hallo</c>, <c>\n</c>,
    /// or control markers like <c>&lt;|turn&gt;</c>. Any position where no vocab prefix
    /// matches falls back to a single-character symbol, which then either participates in
    /// a merge or becomes an UNK during the final piece-to-id lookup.
    /// </summary>
    private List<string> ApplyBpe(string word)
    {
        if (word.Length == 0)
        {
            return [];
        }

        // Fast path: whole word is an atomic vocab entry.
        if (_pieceToId.ContainsKey(word))
        {
            return [word];
        }

        // Greedy longest-prefix segmentation against the vocabulary.
        var symbols = new List<string>();
        var pos = 0;

        while (pos < word.Length)
        {
            var bestLen = 0;

            for (var len = word.Length - pos; len > 0; len--)
            {
                if (_pieceToId.ContainsKey(word.Substring(pos, len)))
                {
                    bestLen = len;
                    break;
                }
            }

            if (bestLen > 0)
            {
                symbols.Add(word.Substring(pos, bestLen));
                pos += bestLen;
            }
            else
            {
                symbols.Add(word[pos].ToString());
                pos++;
            }
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
