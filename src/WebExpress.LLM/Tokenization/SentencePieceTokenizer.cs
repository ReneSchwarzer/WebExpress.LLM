using System;
using System.Collections.Generic;
using System.IO;
using System.Text;

namespace WebExpress.LLM.Tokenization;

/// <summary>
/// A native C# implementation of a SentencePiece tokenizer that provides BPE (Byte Pair Encoding)
/// tokenization compatible with models expecting SentencePiece token IDs.
/// </summary>
/// <remarks>
/// This tokenizer loads vocabulary and merge rules from a SentencePiece model file (.model)
/// by parsing the embedded protobuf data. It supports:
/// <list type="bullet">
///   <item>BPE-based encoding using iterative pair merging</item>
///   <item>Special token handling (BOS, EOS, UNK, PAD)</item>
///   <item>The SentencePiece whitespace convention (▁ prefix for word-initial tokens)</item>
///   <item>Full decode back to original text</item>
/// </list>
/// No external libraries are required.
/// </remarks>
public sealed class SentencePieceTokenizer : ITokenizer
{
    /// <summary>
    /// The Unicode character used by SentencePiece to represent a space/word boundary.
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
    /// Initializes a new instance of the <see cref="SentencePieceTokenizer"/> class with the specified
    /// vocabulary, merge rules, and token configuration.
    /// </summary>
    /// <param name="vocabulary">The mapping from piece strings to token IDs.</param>
    /// <param name="merges">
    /// Ordered list of BPE merge rules. Each entry is a pair of pieces that should be merged,
    /// with earlier entries having higher priority.
    /// </param>
    /// <param name="unknownTokenId">The ID for unknown tokens. Default is 0.</param>
    /// <param name="bosTokenId">The ID for the beginning-of-sequence token. Default is 1.</param>
    /// <param name="eosTokenId">The ID for the end-of-sequence token. Default is 2.</param>
    /// <param name="addBosToken">Whether to prepend a BOS token during encoding. Default is true.</param>
    /// <param name="addEosToken">Whether to append an EOS token during encoding. Default is false.</param>
    public SentencePieceTokenizer(
        IReadOnlyDictionary<string, int> vocabulary,
        IReadOnlyList<(string Left, string Right)> merges,
        int unknownTokenId = 0,
        int bosTokenId = 1,
        int eosTokenId = 2,
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

        var tokenIds = new List<int>();

        if (_addBosToken)
        {
            tokenIds.Add(_bosTokenId);
        }

        // Split text into words preserving whitespace as SentencePiece prefix
        var words = SplitIntoWords(text);

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
    /// Creates a <see cref="SentencePieceTokenizer"/> from a SentencePiece .model file and
    /// an optional <see cref="TokenizerConfiguration"/>.
    /// </summary>
    /// <param name="modelPath">Path to the SentencePiece .model file.</param>
    /// <param name="config">
    /// Optional tokenizer configuration that overrides default special token behavior.
    /// </param>
    /// <returns>A new <see cref="SentencePieceTokenizer"/> instance.</returns>
    public static SentencePieceTokenizer FromModel(string modelPath, TokenizerConfiguration config = null)
    {
        if (string.IsNullOrWhiteSpace(modelPath))
        {
            throw new ArgumentException("Model path must be provided.", nameof(modelPath));
        }

        if (!File.Exists(modelPath))
        {
            throw new FileNotFoundException("SentencePiece model file was not found.", modelPath);
        }

        var modelData = File.ReadAllBytes(modelPath);
        var (vocabulary, merges) = ParseSentencePieceModel(modelData);

        var unkId = vocabulary.TryGetValue("<unk>", out var uid) ? uid : 0;
        var bosId = vocabulary.TryGetValue("<s>", out var bid) ? bid : 1;
        var eosId = vocabulary.TryGetValue("</s>", out var eid) ? eid : 2;

        var addBos = config?.AddBosToken ?? true;
        var addEos = config?.AddEosToken ?? false;

        return new SentencePieceTokenizer(vocabulary, merges, unkId, bosId, eosId, addBos, addEos);
    }

    /// <summary>
    /// Creates a <see cref="SentencePieceTokenizer"/> from an explicit vocabulary and merge list.
    /// This factory method allows creating a tokenizer without a .model file, useful for testing
    /// or when vocabulary data comes from another source.
    /// </summary>
    /// <param name="vocabulary">Mapping of piece strings to token IDs.</param>
    /// <param name="merges">Ordered list of BPE merge rules as space-separated strings (e.g. "a b").</param>
    /// <param name="config">Optional tokenizer configuration for special token settings.</param>
    /// <returns>A new <see cref="SentencePieceTokenizer"/> instance.</returns>
    public static SentencePieceTokenizer FromVocabularyAndMerges(
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

        var unkId = vocabulary.TryGetValue("<unk>", out var uid) ? uid : 0;
        var bosId = vocabulary.TryGetValue("<s>", out var bid) ? bid : 1;
        var eosId = vocabulary.TryGetValue("</s>", out var eid) ? eid : 2;

        var addBos = config?.AddBosToken ?? true;
        var addEos = config?.AddEosToken ?? false;

        return new SentencePieceTokenizer(vocabulary, mergePairs, unkId, bosId, eosId, addBos, addEos);
    }

    /// <summary>
    /// Splits input text into words following SentencePiece conventions.
    /// Every word is prefixed with the ▁ (U+2581) character to represent a word boundary,
    /// including the very first word (matching standard SentencePiece behavior).
    /// </summary>
    private static List<string> SplitIntoWords(string text)
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

    /// <summary>
    /// Parses a SentencePiece protobuf model file to extract vocabulary and merge rules.
    /// This is a minimal protobuf parser that handles the SentencePiece ModelProto format
    /// without requiring a full protobuf library.
    /// </summary>
    internal static (Dictionary<string, int> Vocabulary, List<(string Left, string Right)> Merges)
        ParseSentencePieceModel(byte[] data)
    {
        var vocabulary = new Dictionary<string, int>();
        var merges = new List<(string, string)>();
        var pieces = new List<(string Piece, float Score, int Type)>();

        var offset = 0;

        while (offset < data.Length)
        {
            var (fieldNumber, wireType) = ReadTag(data, ref offset);

            switch (wireType)
            {
                case 0: // Varint
                    ReadVarint(data, ref offset);
                    break;

                case 1: // 64-bit
                    offset += 8;
                    break;

                case 2: // Length-delimited
                    var length = (int)ReadVarint(data, ref offset);

                    if (fieldNumber == 1) // pieces field in ModelProto
                    {
                        var piece = ParseSentencePiece(data, offset, length);
                        pieces.Add(piece);
                    }

                    offset += length;
                    break;

                case 5: // 32-bit
                    offset += 4;
                    break;

                default:
                    // Skip unknown wire types by advancing one byte to try recovery
                    offset++;
                    break;
            }
        }

        // Build vocabulary from pieces
        for (var i = 0; i < pieces.Count; i++)
        {
            var (piece, _, _) = pieces[i];

            if (!vocabulary.ContainsKey(piece))
            {
                vocabulary[piece] = i;
            }
        }

        // Build merge rules: for BPE, each multi-character piece that can be decomposed
        // into two sub-pieces that exist in the vocabulary represents a merge rule.
        // Pieces are ordered by score (higher score = higher priority = lower rank index).
        for (var i = 0; i < pieces.Count; i++)
        {
            var (piece, _, type) = pieces[i];

            // Only process normal tokens (type 1 = UNKNOWN, type 2 = CONTROL, type 3 = USER_DEFINED)
            if (type != 0 || piece.Length < 2)
            {
                continue;
            }

            // Try to find a valid split point
            for (var splitPos = 1; splitPos < piece.Length; splitPos++)
            {
                var left = piece[..splitPos];
                var right = piece[splitPos..];

                if (vocabulary.ContainsKey(left) && vocabulary.ContainsKey(right))
                {
                    merges.Add((left, right));
                    break;
                }
            }
        }

        return (vocabulary, merges);
    }

    /// <summary>
    /// Parses a single SentencePiece entry from the protobuf data.
    /// </summary>
    private static (string Piece, float Score, int Type) ParseSentencePiece(
        byte[] data, int start, int length)
    {
        var piece = string.Empty;
        var score = 0.0f;
        var type = 0;
        var offset = start;
        var end = start + length;

        while (offset < end)
        {
            var (fieldNumber, wireType) = ReadTag(data, ref offset);

            switch (fieldNumber)
            {
                case 1 when wireType == 2: // piece string
                    var strLen = (int)ReadVarint(data, ref offset);
                    piece = Encoding.UTF8.GetString(data, offset, strLen);
                    offset += strLen;
                    break;

                case 2 when wireType == 5: // score (float)
                    score = BitConverter.ToSingle(data, offset);
                    offset += 4;
                    break;

                case 3 when wireType == 0: // type (enum/varint)
                    type = (int)ReadVarint(data, ref offset);
                    break;

                default:
                    SkipField(data, wireType, ref offset);
                    break;
            }
        }

        return (piece, score, type);
    }

    /// <summary>
    /// Reads a protobuf tag (field number + wire type) from the data.
    /// </summary>
    private static (int FieldNumber, int WireType) ReadTag(byte[] data, ref int offset)
    {
        var tag = (int)ReadVarint(data, ref offset);

        return (tag >> 3, tag & 0x07);
    }

    /// <summary>
    /// Reads a variable-length encoded integer from the protobuf data.
    /// </summary>
    private static ulong ReadVarint(byte[] data, ref int offset)
    {
        ulong result = 0;
        var shift = 0;

        while (offset < data.Length)
        {
            var b = data[offset++];
            result |= (ulong)(b & 0x7F) << shift;

            if ((b & 0x80) == 0)
            {
                break;
            }

            shift += 7;
        }

        return result;
    }

    /// <summary>
    /// Skips a protobuf field based on its wire type.
    /// </summary>
    private static void SkipField(byte[] data, int wireType, ref int offset)
    {
        switch (wireType)
        {
            case 0: // Varint
                ReadVarint(data, ref offset);
                break;

            case 1: // 64-bit
                offset += 8;
                break;

            case 2: // Length-delimited
                var length = (int)ReadVarint(data, ref offset);
                offset += length;
                break;

            case 5: // 32-bit
                offset += 4;
                break;
        }
    }
}
