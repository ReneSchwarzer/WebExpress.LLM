using WebExpress.LLM.Tokenization;

namespace WebExpress.LLM.Test.Tokenization;

public sealed class GemmaTokenizerTests
{
    /// <summary>
    /// Creates a simple test vocabulary and merge list for Gemma BPE testing.
    /// Uses Gemma-style special tokens: &lt;unk&gt;=0, &lt;bos&gt;=1, &lt;eos&gt;=2
    /// </summary>
    private static (Dictionary<string, int> Vocabulary, List<(string, string)> Merges) CreateTestVocabulary()
    {
        var vocab = new Dictionary<string, int>
        {
            ["<unk>"] = 0,
            ["<bos>"] = 1,
            ["<eos>"] = 2,
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
            ("h", "e"),
            ("l", "l"),
            ("l", "o"),
            ("\u2581", "h"),
            ("ll", "o"),
            ("\u2581h", "e"),
            ("he", "l"),
            ("\u2581he", "l"),
            ("e", "llo"),
            ("h", "ello"),
            ("\u2581", "hello"),
            ("\u2581hel", "lo"),
            ("\u2581", "w"),
            ("o", "r"),
            ("\u2581w", "o"),
            ("r", "ld"),
            ("or", "ld"),
            ("w", "orld"),
            ("\u2581wo", "rld"),
        };

        return (vocab, merges);
    }

    [Fact]
    public void Constructor_WithEmptyVocabulary_ShouldThrow()
    {
        var emptyVocab = new Dictionary<string, int>();
        var merges = new List<(string, string)>();

        Assert.Throws<ArgumentException>(() =>
            new GemmaTokenizer(emptyVocab, merges));
    }

    [Fact]
    public void Constructor_WithNullVocabulary_ShouldThrow()
    {
        Assert.Throws<ArgumentNullException>(() =>
            new GemmaTokenizer(null, new List<(string, string)>()));
    }

    [Fact]
    public void Constructor_WithNullMerges_ShouldThrow()
    {
        var vocab = new Dictionary<string, int> { ["a"] = 0 };

        Assert.Throws<ArgumentNullException>(() =>
            new GemmaTokenizer(vocab, null));
    }

    [Fact]
    public void Encode_WithNull_ShouldThrow()
    {
        var (vocab, merges) = CreateTestVocabulary();
        var tokenizer = new GemmaTokenizer(vocab, merges, addBosToken: false);

        Assert.Throws<ArgumentNullException>(() => tokenizer.Encode(null));
    }

    [Fact]
    public void Encode_WithEmptyString_ShouldReturnBosOnly()
    {
        var (vocab, merges) = CreateTestVocabulary();
        var tokenizer = new GemmaTokenizer(vocab, merges, bosTokenId: 1, addBosToken: true, addEosToken: false);

        var tokens = tokenizer.Encode("");

        Assert.Single(tokens);
        Assert.Equal(1, tokens[0]); // BOS token
    }

    [Fact]
    public void Encode_WithEmptyStringAndNoSpecialTokens_ShouldReturnEmpty()
    {
        var (vocab, merges) = CreateTestVocabulary();
        var tokenizer = new GemmaTokenizer(vocab, merges, addBosToken: false, addEosToken: false);

        var tokens = tokenizer.Encode("");

        Assert.Empty(tokens);
    }

    [Fact]
    public void Encode_WithBosAndEos_ShouldWrapTokens()
    {
        var vocab = new Dictionary<string, int>
        {
            ["<unk>"] = 0,
            ["<bos>"] = 1,
            ["<eos>"] = 2,
            ["\u2581a"] = 3
        };

        var merges = new List<(string, string)>
        {
            ("\u2581", "a")
        };

        var tokenizer = new GemmaTokenizer(vocab, merges,
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
        var tokenizer = new GemmaTokenizer(vocab, merges, addBosToken: false, addEosToken: false);

        var tokens = tokenizer.Encode("hello");

        Assert.Contains(21, tokens);
    }

    [Fact]
    public void Encode_UnknownCharacter_ShouldUseUnknownTokenId()
    {
        var vocab = new Dictionary<string, int>
        {
            ["<unk>"] = 0,
            ["<bos>"] = 1,
            ["<eos>"] = 2,
            ["\u2581a"] = 3
        };

        var merges = new List<(string, string)>
        {
            ("\u2581", "a")
        };

        var tokenizer = new GemmaTokenizer(vocab, merges,
            unknownTokenId: 0, addBosToken: false, addEosToken: false);

        var tokens = tokenizer.Encode("z");

        Assert.Contains(0, tokens);
    }

    [Fact]
    public void Decode_WithNull_ShouldThrow()
    {
        var (vocab, merges) = CreateTestVocabulary();
        var tokenizer = new GemmaTokenizer(vocab, merges, addBosToken: false);

        Assert.Throws<ArgumentNullException>(() => tokenizer.Decode(null));
    }

    [Fact]
    public void Decode_ShouldRemoveSpaceSymbol()
    {
        var (vocab, merges) = CreateTestVocabulary();
        var tokenizer = new GemmaTokenizer(vocab, merges, addBosToken: false);

        var text = tokenizer.Decode(new[] { 21 });

        Assert.Equal("hello", text);
    }

    [Fact]
    public void Decode_MultipleWords_ShouldProduceCorrectSpaces()
    {
        var (vocab, merges) = CreateTestVocabulary();
        var tokenizer = new GemmaTokenizer(vocab, merges, addBosToken: false);

        var text = tokenizer.Decode(new[] { 21, 28 });

        Assert.Equal("hello world", text);
    }

    [Fact]
    public void Decode_ShouldSkipBosAndEos()
    {
        var (vocab, merges) = CreateTestVocabulary();
        var tokenizer = new GemmaTokenizer(vocab, merges, bosTokenId: 1, eosTokenId: 2, addBosToken: false);

        var text = tokenizer.Decode(new[] { 1, 21, 28, 2 });

        Assert.Equal("hello world", text);
    }

    [Fact]
    public void Decode_UnknownTokenId_ShouldOutputUnkMarker()
    {
        var (vocab, merges) = CreateTestVocabulary();
        var tokenizer = new GemmaTokenizer(vocab, merges, addBosToken: false);

        var text = tokenizer.Decode(new[] { 999 });

        Assert.Equal("<unk>", text);
    }

    [Fact]
    public void EncodeDecode_RoundTrip_ShouldBeReversible()
    {
        var (vocab, merges) = CreateTestVocabulary();
        var tokenizer = new GemmaTokenizer(vocab, merges, addBosToken: false, addEosToken: false);

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
            ["<bos>"] = 1,
            ["<eos>"] = 2,
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
            "bos_token": "<bos>",
            "eos_token": "<eos>",
            "add_bos_token": false,
            "add_eos_token": false
        }
        """);

        var tokenizer = GemmaTokenizer.FromVocabularyAndMerges(vocab, mergeStrings, config);

        var tokens = tokenizer.Encode("ab");

        Assert.Contains(8, tokens);
    }

    [Fact]
    public void FromVocabularyAndMerges_WithConfigBosEos_ShouldRespectConfig()
    {
        var vocab = new Dictionary<string, int>
        {
            ["<unk>"] = 0,
            ["<bos>"] = 1,
            ["<eos>"] = 2,
            ["\u2581a"] = 3
        };

        var mergeStrings = new List<string> { "\u2581 a" };

        var config = TokenizerConfiguration.FromJson("""
        {
            "bos_token": "<bos>",
            "eos_token": "<eos>",
            "add_bos_token": true,
            "add_eos_token": true
        }
        """);

        var tokenizer = GemmaTokenizer.FromVocabularyAndMerges(vocab, mergeStrings, config);

        var tokens = tokenizer.Encode("a");

        Assert.Equal(1, tokens[0]);  // BOS
        Assert.Equal(2, tokens[^1]); // EOS
    }

    [Fact]
    public void FromTokenizerJson_WithNullPath_ShouldThrow()
    {
        Assert.Throws<ArgumentException>(() => GemmaTokenizer.FromTokenizerJson(null));
    }

    [Fact]
    public void FromTokenizerJson_WithNonexistentFile_ShouldThrow()
    {
        Assert.Throws<FileNotFoundException>(() =>
            GemmaTokenizer.FromTokenizerJson("/nonexistent/tokenizer.json"));
    }

    [Fact]
    public void FromTokenizerJson_WithMissingModelSection_ShouldThrow()
    {
        var tempFile = Path.GetTempFileName();

        try
        {
            File.WriteAllText(tempFile, """{ "version": "1.0" }""");
            Assert.Throws<InvalidDataException>(() => GemmaTokenizer.FromTokenizerJson(tempFile));
        }
        finally
        {
            File.Delete(tempFile);
        }
    }

    [Fact]
    public void FromTokenizerJson_WithValidJson_ShouldCreateFunctionalTokenizer()
    {
        var tempFile = Path.GetTempFileName();

        try
        {
            var json = """
            {
                "version": "1.0",
                "model": {
                    "type": "BPE",
                    "unk_token": "<unk>",
                    "vocab": {
                        "<unk>": 0,
                        "<bos>": 1,
                        "<eos>": 2,
                        "\u2581": 3,
                        "a": 4,
                        "b": 5,
                        "ab": 6,
                        "\u2581a": 7,
                        "\u2581ab": 8
                    },
                    "merges": [
                        "a b",
                        "\u2581 a",
                        "\u2581a b"
                    ]
                }
            }
            """;

            File.WriteAllText(tempFile, json);

            var config = TokenizerConfiguration.FromJson("""
            {
                "bos_token": "<bos>",
                "eos_token": "<eos>",
                "add_bos_token": false,
                "add_eos_token": false
            }
            """);
            var tokenizer = GemmaTokenizer.FromTokenizerJson(tempFile, config);
            var tokens = tokenizer.Encode("ab");

            Assert.Contains(8, tokens);
        }
        finally
        {
            File.Delete(tempFile);
        }
    }

    [Fact]
    public void FromTokenizerJson_WithArrayMerges_ShouldCreateFunctionalTokenizer()
    {
        var tempFile = Path.GetTempFileName();

        try
        {
            // Simulates Gemma-style tokenizer.json where merges are arrays ["left", "right"],
            // not space-separated strings "left right".
            var json = """
            {
                "version": "1.0",
                "model": {
                    "type": "BPE",
                    "unk_token": "<unk>",
                    "vocab": {
                        "<unk>": 0,
                        "<bos>": 1,
                        "<eos>": 2,
                        "\u2581": 3,
                        "a": 4,
                        "b": 5,
                        "ab": 6,
                        "\u2581a": 7,
                        "\u2581ab": 8
                    },
                    "merges": [
                        ["a", "b"],
                        ["\u2581", "a"],
                        ["\u2581a", "b"]
                    ]
                }
            }
            """;

            File.WriteAllText(tempFile, json);

            var config = TokenizerConfiguration.FromJson("""
            {
                "bos_token": "<bos>",
                "eos_token": "<eos>",
                "unk_token": "<unk>",
                "add_bos_token": false,
                "add_eos_token": false
            }
            """);

            var tokenizer = GemmaTokenizer.FromTokenizerJson(tempFile, config);
            var tokens = tokenizer.Encode("ab");

            Assert.Contains(8, tokens);
        }
        finally
        {
            File.Delete(tempFile);
        }
    }

    [Fact]
    public void FromTokenizerJson_WithCustomBosEosTokenNames_ShouldResolveFromConfig()
    {
        var tempFile = Path.GetTempFileName();

        try
        {
            var json = """
            {
                "version": "1.0",
                "model": {
                    "type": "BPE",
                    "unk_token": "<unk>",
                    "vocab": {
                        "<unk>": 0,
                        "<bos>": 2,
                        "<eos>": 1,
                        "\u2581a": 3
                    },
                    "merges": [ "\u2581 a" ]
                }
            }
            """;

            File.WriteAllText(tempFile, json);

            var config = TokenizerConfiguration.FromJson("""
            {
                "bos_token": "<bos>",
                "eos_token": "<eos>",
                "unk_token": "<unk>",
                "add_bos_token": true,
                "add_eos_token": true
            }
            """);

            var tokenizer = GemmaTokenizer.FromTokenizerJson(tempFile, config);
            var tokens = tokenizer.Encode("a");

            // BOS should be 2 (from "<bos>" in vocab), EOS should be 1 (from "<eos>" in vocab)
            Assert.Equal(2, tokens[0]);   // <bos>
            Assert.Equal(1, tokens[^1]);  // <eos>
        }
        finally
        {
            File.Delete(tempFile);
        }
    }

    [Fact]
    public void FromTokenizerJson_WithEmptyVocab_ShouldThrow()
    {
        var tempFile = Path.GetTempFileName();

        try
        {
            File.WriteAllText(tempFile, """{ "model": { "type": "BPE", "vocab": {}, "merges": [] } }""");
            Assert.Throws<InvalidDataException>(() => GemmaTokenizer.FromTokenizerJson(tempFile));
        }
        finally
        {
            File.Delete(tempFile);
        }
    }

    [Fact]
    public void FromVocabularyAndMerges_WithConfigBosEosTokenNames_ShouldResolveCorrectIds()
    {
        var vocab = new Dictionary<string, int>
        {
            ["<unk>"] = 0,
            ["<bos>"] = 2,
            ["<eos>"] = 1,
            ["\u2581a"] = 3
        };

        var mergeStrings = new List<string> { "\u2581 a" };

        var config = TokenizerConfiguration.FromJson("""
        {
            "bos_token": "<bos>",
            "eos_token": "<eos>",
            "unk_token": "<unk>",
            "add_bos_token": true,
            "add_eos_token": true
        }
        """);

        var tokenizer = GemmaTokenizer.FromVocabularyAndMerges(vocab, mergeStrings, config);
        var tokens = tokenizer.Encode("a");

        Assert.Equal(2, tokens[0]);   // <bos>
        Assert.Equal(1, tokens[^1]);  // <eos>
    }

    [Fact]
    public void Normalize_ShouldApplyNfkc()
    {
        // ﬁ (U+FB01 LATIN SMALL LIGATURE FI) should normalize to "fi"
        var result = GemmaTokenizer.Normalize("\uFB01");

        Assert.Equal("fi", result);
    }

    [Fact]
    public void Normalize_AsciiText_ShouldBeUnchanged()
    {
        var result = GemmaTokenizer.Normalize("hello world");

        Assert.Equal("hello world", result);
    }

    [Fact]
    public void PreTokenize_ShouldSplitOnWhitespace()
    {
        var words = GemmaTokenizer.PreTokenize("hello world");

        Assert.Equal(2, words.Count);
        Assert.Equal("\u2581hello", words[0]);
        Assert.Equal("\u2581world", words[1]);
    }

    [Fact]
    public void PreTokenize_SingleWord_ShouldPrefixWithSpaceSymbol()
    {
        var words = GemmaTokenizer.PreTokenize("hello");

        Assert.Single(words);
        Assert.Equal("\u2581hello", words[0]);
    }

    [Fact]
    public void PreTokenize_EmptyString_ShouldReturnEmpty()
    {
        var words = GemmaTokenizer.PreTokenize("");

        Assert.Empty(words);
    }

    [Fact]
    public void SpaceSymbol_ShouldBeCorrectUnicodeCharacter()
    {
        Assert.Equal('\u2581', GemmaTokenizer.SpaceSymbol);
    }
}
