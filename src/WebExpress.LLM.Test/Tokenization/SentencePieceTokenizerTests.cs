using WebExpress.LLM.Tokenization;

namespace WebExpress.LLM.Test.Tokenization;

public sealed class SentencePieceTokenizerTests
{
    /// <summary>
    /// Creates a simple test vocabulary and merge list for BPE testing.
    /// Vocabulary: ▁, h, e, l, o, w, r, d, he, lo, ▁he, llo, ▁hel, ▁hello,
    ///             ▁w, or, ▁wo, rld, ▁wor, ▁world
    /// Plus special tokens: &lt;unk&gt;=0, &lt;s&gt;=1, &lt;/s&gt;=2
    /// </summary>
    private static (Dictionary<string, int> Vocabulary, List<(string, string)> Merges) CreateTestVocabulary()
    {
        var vocab = new Dictionary<string, int>
        {
            ["<unk>"] = 0,
            ["<s>"] = 1,
            ["</s>"] = 2,
            ["\u2581"] = 3,
            ["h"] = 4,
            ["e"] = 5,
            ["l"] = 6,
            ["o"] = 7,
            ["w"] = 8,
            ["r"] = 9,
            ["d"] = 10,
            ["he"] = 11,
            ["lo"] = 12,
            ["ll"] = 13,
            ["\u2581h"] = 14,
            ["llo"] = 15,
            ["\u2581he"] = 16,
            ["hel"] = 17,
            ["\u2581hel"] = 18,
            ["ello"] = 19,
            ["hello"] = 20,
            ["\u2581hello"] = 21,
            ["\u2581w"] = 22,
            ["or"] = 23,
            ["\u2581wo"] = 24,
            ["rld"] = 25,
            ["orld"] = 26,
            ["world"] = 27,
            ["\u2581world"] = 28
        };

        // Merges in priority order
        var merges = new List<(string, string)>
        {
            ("h", "e"),          // h + e → he
            ("l", "l"),          // l + l → ll
            ("l", "o"),          // l + o → lo
            ("\u2581", "h"),     // ▁ + h → ▁h
            ("ll", "o"),         // ll + o → llo
            ("\u2581h", "e"),    // ▁h + e → ▁he
            ("he", "l"),         // he + l → hel
            ("\u2581he", "l"),   // ▁he + l → ▁hel
            ("e", "llo"),        // e + llo → ello
            ("h", "ello"),       // h + ello → hello
            ("\u2581", "hello"), // ▁ + hello → ▁hello (not used since ▁hel + lo is more typical)
            ("\u2581hel", "lo"), // ▁hel + lo → ▁hello (this is actually how it merges)
            ("\u2581", "w"),     // ▁ + w → ▁w
            ("o", "r"),          // o + r → or
            ("\u2581w", "o"),    // ▁w + o → ▁wo
            ("r", "ld"),         // r + ld → rld (this won't apply since ld isn't in vocab, so use or + ld)
            ("or", "ld"),        // or + ld → orld (won't happen unless ld is a piece)
            ("w", "orld"),       // w + orld → world
            ("\u2581wo", "rld"), // ▁wo + rld → ▁world
        };

        return (vocab, merges);
    }

    [Fact]
    public void Constructor_WithEmptyVocabulary_ShouldThrow()
    {
        var emptyVocab = new Dictionary<string, int>();
        var merges = new List<(string, string)>();

        Assert.Throws<ArgumentException>(() =>
            new SentencePieceTokenizer(emptyVocab, merges));
    }

    [Fact]
    public void Constructor_WithNullVocabulary_ShouldThrow()
    {
        Assert.Throws<ArgumentNullException>(() =>
            new SentencePieceTokenizer(null, new List<(string, string)>()));
    }

    [Fact]
    public void Constructor_WithNullMerges_ShouldThrow()
    {
        var vocab = new Dictionary<string, int> { ["a"] = 0 };

        Assert.Throws<ArgumentNullException>(() =>
            new SentencePieceTokenizer(vocab, null));
    }

    [Fact]
    public void Encode_WithNull_ShouldThrow()
    {
        var (vocab, merges) = CreateTestVocabulary();
        var tokenizer = new SentencePieceTokenizer(vocab, merges, addBosToken: false);

        Assert.Throws<ArgumentNullException>(() => tokenizer.Encode(null));
    }

    [Fact]
    public void Encode_WithEmptyString_ShouldReturnBosOnly()
    {
        var (vocab, merges) = CreateTestVocabulary();
        var tokenizer = new SentencePieceTokenizer(vocab, merges, bosTokenId: 1, addBosToken: true, addEosToken: false);

        var tokens = tokenizer.Encode("");

        Assert.Single(tokens);
        Assert.Equal(1, tokens[0]); // BOS token
    }

    [Fact]
    public void Encode_WithEmptyStringAndNoSpecialTokens_ShouldReturnEmpty()
    {
        var (vocab, merges) = CreateTestVocabulary();
        var tokenizer = new SentencePieceTokenizer(vocab, merges, addBosToken: false, addEosToken: false);

        var tokens = tokenizer.Encode("");

        Assert.Empty(tokens);
    }

    [Fact]
    public void Encode_WithBosAndEos_ShouldWrapTokens()
    {
        var vocab = new Dictionary<string, int>
        {
            ["<unk>"] = 0,
            ["<s>"] = 1,
            ["</s>"] = 2,
            ["\u2581a"] = 3
        };

        var merges = new List<(string, string)>
        {
            ("\u2581", "a")
        };

        var tokenizer = new SentencePieceTokenizer(vocab, merges,
            bosTokenId: 1, eosTokenId: 2, addBosToken: true, addEosToken: true);

        var tokens = tokenizer.Encode("a");

        Assert.Equal(1, tokens[0]);                // BOS
        Assert.Equal(3, tokens[1]);                // ▁a
        Assert.Equal(2, tokens[^1]);               // EOS
    }

    [Fact]
    public void Encode_SimpleWord_ShouldApplyBpeMerges()
    {
        var (vocab, merges) = CreateTestVocabulary();
        var tokenizer = new SentencePieceTokenizer(vocab, merges, addBosToken: false, addEosToken: false);

        var tokens = tokenizer.Encode("hello");

        // "hello" → "▁hello" which should be looked up as a single piece (ID 21)
        Assert.Contains(21, tokens);
    }

    [Fact]
    public void Encode_UnknownCharacter_ShouldUseUnknownTokenId()
    {
        var vocab = new Dictionary<string, int>
        {
            ["<unk>"] = 0,
            ["<s>"] = 1,
            ["</s>"] = 2,
            ["\u2581a"] = 3
        };

        var merges = new List<(string, string)>
        {
            ("\u2581", "a")
        };

        var tokenizer = new SentencePieceTokenizer(vocab, merges,
            unknownTokenId: 0, addBosToken: false, addEosToken: false);

        var tokens = tokenizer.Encode("z");

        // 'z' is not in vocabulary, should map to unknown
        Assert.Contains(0, tokens);
    }

    [Fact]
    public void Decode_WithNull_ShouldThrow()
    {
        var (vocab, merges) = CreateTestVocabulary();
        var tokenizer = new SentencePieceTokenizer(vocab, merges, addBosToken: false);

        Assert.Throws<ArgumentNullException>(() => tokenizer.Decode(null));
    }

    [Fact]
    public void Decode_ShouldRemoveSpaceSymbol()
    {
        var (vocab, merges) = CreateTestVocabulary();
        var tokenizer = new SentencePieceTokenizer(vocab, merges, addBosToken: false);

        // Decode ▁hello (ID 21) → "hello" (leading space from ▁ is stripped)
        var text = tokenizer.Decode(new[] { 21 });

        Assert.Equal("hello", text);
    }

    [Fact]
    public void Decode_MultipleWords_ShouldProduceCorrectSpaces()
    {
        var (vocab, merges) = CreateTestVocabulary();
        var tokenizer = new SentencePieceTokenizer(vocab, merges, addBosToken: false);

        // Decode ▁hello (21) + ▁world (28) → "hello world"
        var text = tokenizer.Decode(new[] { 21, 28 });

        Assert.Equal("hello world", text);
    }

    [Fact]
    public void Decode_ShouldSkipBosAndEos()
    {
        var (vocab, merges) = CreateTestVocabulary();
        var tokenizer = new SentencePieceTokenizer(vocab, merges, bosTokenId: 1, eosTokenId: 2, addBosToken: false);

        // Including BOS(1) and EOS(2) should not affect output
        var text = tokenizer.Decode(new[] { 1, 21, 28, 2 });

        Assert.Equal("hello world", text);
    }

    [Fact]
    public void Decode_UnknownTokenId_ShouldOutputUnkMarker()
    {
        var (vocab, merges) = CreateTestVocabulary();
        var tokenizer = new SentencePieceTokenizer(vocab, merges, addBosToken: false);

        var text = tokenizer.Decode(new[] { 999 });

        Assert.Equal("<unk>", text);
    }

    [Fact]
    public void EncodeDecode_RoundTrip_ShouldBeReversible()
    {
        var (vocab, merges) = CreateTestVocabulary();
        var tokenizer = new SentencePieceTokenizer(vocab, merges, addBosToken: false, addEosToken: false);

        var tokens = tokenizer.Encode("hello");
        var decoded = tokenizer.Decode(tokens);

        Assert.Equal("hello", decoded);
    }

    [Fact]
    public void FromVocabularyAndMerges_ShouldCreateFunctionalTokenizer()
    {
        var vocab = new Dictionary<string, int>
        {
            ["<unk>"] = 0,
            ["<s>"] = 1,
            ["</s>"] = 2,
            ["\u2581"] = 3,
            ["a"] = 4,
            ["b"] = 5,
            ["ab"] = 6,
            ["\u2581a"] = 7,
            ["\u2581ab"] = 8
        };

        var mergeStrings = new List<string>
        {
            "a b",
            "\u2581 a",
            "\u2581a b"
        };

        var config = TokenizerConfiguration.FromJson("""
        {
            "add_bos_token": false,
            "add_eos_token": false
        }
        """);

        var tokenizer = SentencePieceTokenizer.FromVocabularyAndMerges(vocab, mergeStrings, config);

        var tokens = tokenizer.Encode("ab");

        // Should encode "ab" → "▁ab" which is ID 8
        Assert.Contains(8, tokens);
    }

    [Fact]
    public void FromVocabularyAndMerges_WithConfigBosEos_ShouldRespectConfig()
    {
        var vocab = new Dictionary<string, int>
        {
            ["<unk>"] = 0,
            ["<s>"] = 1,
            ["</s>"] = 2,
            ["\u2581a"] = 3
        };

        var mergeStrings = new List<string> { "\u2581 a" };

        var config = TokenizerConfiguration.FromJson("""
        {
            "add_bos_token": true,
            "add_eos_token": true
        }
        """);

        var tokenizer = SentencePieceTokenizer.FromVocabularyAndMerges(vocab, mergeStrings, config);

        var tokens = tokenizer.Encode("a");

        Assert.Equal(1, tokens[0]);  // BOS
        Assert.Equal(2, tokens[^1]); // EOS
    }

    [Fact]
    public void FromModel_WithNullPath_ShouldThrow()
    {
        Assert.Throws<ArgumentException>(() => SentencePieceTokenizer.FromModel(null));
    }

    [Fact]
    public void FromModel_WithNonexistentFile_ShouldThrow()
    {
        Assert.Throws<FileNotFoundException>(() =>
            SentencePieceTokenizer.FromModel("/nonexistent/model.model"));
    }

    [Fact]
    public void ParseSentencePieceModel_WithMinimalProtobuf_ShouldExtractPieces()
    {
        // Build a minimal protobuf with two pieces: "<unk>" (type=2) and "a" (type=0)
        var modelBytes = BuildMinimalSentencePieceModel(new[]
        {
            ("<unk>", 0.0f, 2),
            ("<s>", 0.0f, 2),
            ("</s>", 0.0f, 2),
            ("a", 0.0f, 0),
            ("b", 0.0f, 0),
            ("ab", -1.0f, 0)
        });

        var (vocabulary, merges) = SentencePieceTokenizer.ParseSentencePieceModel(modelBytes);

        Assert.True(vocabulary.ContainsKey("<unk>"));
        Assert.True(vocabulary.ContainsKey("a"));
        Assert.True(vocabulary.ContainsKey("b"));
        Assert.True(vocabulary.ContainsKey("ab"));
        Assert.Equal(0, vocabulary["<unk>"]);
        Assert.Equal(3, vocabulary["a"]);
        Assert.Equal(4, vocabulary["b"]);
        Assert.Equal(5, vocabulary["ab"]);
    }

    [Fact]
    public void ParseSentencePieceModel_ShouldGenerateMergesForMultiCharPieces()
    {
        var modelBytes = BuildMinimalSentencePieceModel(new[]
        {
            ("<unk>", 0.0f, 2),
            ("a", 0.0f, 0),
            ("b", 0.0f, 0),
            ("ab", -1.0f, 0)
        });

        var (_, merges) = SentencePieceTokenizer.ParseSentencePieceModel(modelBytes);

        // "ab" should generate a merge rule (a, b)
        Assert.Contains(("a", "b"), merges);
    }

    [Fact]
    public void SpaceSymbol_ShouldBeCorrectUnicodeCharacter()
    {
        Assert.Equal('\u2581', SentencePieceTokenizer.SpaceSymbol);
    }

    /// <summary>
    /// Builds a minimal protobuf representation of a SentencePiece model for testing.
    /// Each piece is encoded as a field 1 (length-delimited) in the top-level message,
    /// containing field 1 (string), field 2 (float), and field 3 (varint type).
    /// </summary>
    private static byte[] BuildMinimalSentencePieceModel(
        (string Piece, float Score, int Type)[] pieces)
    {
        var ms = new MemoryStream();

        foreach (var (piece, score, type) in pieces)
        {
            // Build the inner SentencePiece message
            var inner = new MemoryStream();

            // Field 1: piece (string), tag = (1 << 3) | 2 = 0x0A
            var pieceBytes = System.Text.Encoding.UTF8.GetBytes(piece);
            inner.WriteByte(0x0A);
            WriteVarint(inner, (ulong)pieceBytes.Length);
            inner.Write(pieceBytes, 0, pieceBytes.Length);

            // Field 2: score (float), tag = (2 << 3) | 5 = 0x15
            inner.WriteByte(0x15);
            var scoreBytes = BitConverter.GetBytes(score);
            inner.Write(scoreBytes, 0, 4);

            // Field 3: type (varint), tag = (3 << 3) | 0 = 0x18
            inner.WriteByte(0x18);
            WriteVarint(inner, (ulong)type);

            var innerData = inner.ToArray();

            // Write as field 1 (length-delimited) in outer message: tag = (1 << 3) | 2 = 0x0A
            ms.WriteByte(0x0A);
            WriteVarint(ms, (ulong)innerData.Length);
            ms.Write(innerData, 0, innerData.Length);
        }

        return ms.ToArray();
    }

    private static void WriteVarint(Stream stream, ulong value)
    {
        while (value > 0x7F)
        {
            stream.WriteByte((byte)(0x80 | (value & 0x7F)));
            value >>= 7;
        }

        stream.WriteByte((byte)value);
    }
}
