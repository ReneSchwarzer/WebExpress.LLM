using System;
using System.Collections.Generic;
using System.IO;
using System.Text.Json;
using System.Text.Json.Serialization;

namespace WebExpress.LLM.Tokenization;

/// <summary>
/// Represents the configuration of a tokenizer as loaded from a tokenizer_config.json file.
/// Serves as the single source of truth for all tokenizer settings including vocabulary references,
/// normalization, special tokens, and pre/post-processing configuration.
/// </summary>
public sealed class TokenizerConfiguration
{
    /// <summary>
    /// Gets the backend identifier for the tokenizer (e.g. "tokenizers").
    /// </summary>
    [JsonPropertyName("backend")]
    public string Backend { get; init; } = string.Empty;

    /// <summary>
    /// Gets the tokenizer class name (e.g. "GemmaTokenizer", "SentencePieceTokenizer").
    /// </summary>
    [JsonPropertyName("tokenizer_class")]
    public string TokenizerClass { get; init; } = string.Empty;

    /// <summary>
    /// Gets the beginning-of-sequence token.
    /// </summary>
    [JsonPropertyName("bos_token")]
    public string BosToken { get; init; } = string.Empty;

    /// <summary>
    /// Gets the end-of-sequence token.
    /// </summary>
    [JsonPropertyName("eos_token")]
    public string EosToken { get; init; } = string.Empty;

    /// <summary>
    /// Gets the unknown token.
    /// </summary>
    [JsonPropertyName("unk_token")]
    public string UnkToken { get; init; } = string.Empty;

    /// <summary>
    /// Gets the padding token.
    /// </summary>
    [JsonPropertyName("pad_token")]
    public string PadToken { get; init; } = string.Empty;

    /// <summary>
    /// Gets the audio content token.
    /// </summary>
    [JsonPropertyName("audio_token")]
    public string AudioToken { get; init; } = string.Empty;

    /// <summary>
    /// Gets the beginning-of-audio token.
    /// </summary>
    [JsonPropertyName("boa_token")]
    public string BoaToken { get; init; } = string.Empty;

    /// <summary>
    /// Gets the beginning-of-image token.
    /// </summary>
    [JsonPropertyName("boi_token")]
    public string BoiToken { get; init; } = string.Empty;

    /// <summary>
    /// Gets the end-of-audio token.
    /// </summary>
    [JsonPropertyName("eoa_token")]
    public string EoaToken { get; init; } = string.Empty;

    /// <summary>
    /// Gets the end-of-image token.
    /// </summary>
    [JsonPropertyName("eoi_token")]
    public string EoiToken { get; init; } = string.Empty;

    /// <summary>
    /// Gets the maximum model input length in tokens.
    /// HuggingFace configs sometimes store this as a very large floating-point value (e.g. 1e30)
    /// to indicate "unlimited"; such values are clamped to <see cref="long.MaxValue"/>.
    /// </summary>
    [JsonPropertyName("model_max_length")]
    [JsonConverter(typeof(LongFromNumberConverter))]
    public long ModelMaxLength { get; init; }

    /// <summary>
    /// Gets a value indicating whether the tokenizer adds a BOS token during encoding.
    /// </summary>
    [JsonPropertyName("add_bos_token")]
    public bool AddBosToken { get; init; }

    /// <summary>
    /// Gets a value indicating whether the tokenizer adds an EOS token during encoding.
    /// </summary>
    [JsonPropertyName("add_eos_token")]
    public bool AddEosToken { get; init; }

    /// <summary>
    /// Gets additional special tokens that the tokenizer recognizes.
    /// </summary>
    [JsonPropertyName("additional_special_tokens")]
    public IReadOnlyList<string> AdditionalSpecialTokens { get; init; } = [];

    /// <summary>
    /// Gets the SentencePiece model keyword arguments.
    /// </summary>
    [JsonPropertyName("sp_model_kwargs")]
    public Dictionary<string, JsonElement> SpModelKwargs { get; init; } = [];

    /// <summary>
    /// Gets a value indicating whether to clean up tokenization spaces in decoded output.
    /// </summary>
    [JsonPropertyName("clean_up_tokenization_spaces")]
    public bool CleanUpTokenizationSpaces { get; init; }

    /// <summary>
    /// Gets the name or path of the tokenizer/model.
    /// </summary>
    [JsonPropertyName("name_or_path")]
    public string NameOrPath { get; init; } = string.Empty;

    /// <summary>
    /// Loads a tokenizer configuration from the specified JSON file.
    /// </summary>
    /// <param name="filePath">The path to the tokenizer_config.json file.</param>
    /// <returns>A new <see cref="TokenizerConfiguration"/> instance.</returns>
    /// <exception cref="ArgumentException">Thrown when <paramref name="filePath"/> is null or empty.</exception>
    /// <exception cref="FileNotFoundException">Thrown when the specified file does not exist.</exception>
    /// <exception cref="InvalidDataException">Thrown when the file cannot be deserialized.</exception>
    public static TokenizerConfiguration FromFile(string filePath)
    {
        if (string.IsNullOrWhiteSpace(filePath))
        {
            throw new ArgumentException("File path must be provided.", nameof(filePath));
        }

        if (!File.Exists(filePath))
        {
            throw new FileNotFoundException("Tokenizer configuration file was not found.", filePath);
        }

        var json = File.ReadAllText(filePath);

        return FromJson(json);
    }

    /// <summary>
    /// Loads a tokenizer configuration from the specified JSON string.
    /// </summary>
    /// <param name="json">The JSON string representing the tokenizer configuration.</param>
    /// <returns>A new <see cref="TokenizerConfiguration"/> instance.</returns>
    /// <exception cref="ArgumentException">Thrown when <paramref name="json"/> is null or empty.</exception>
    /// <exception cref="InvalidDataException">Thrown when the JSON cannot be deserialized.</exception>
    public static TokenizerConfiguration FromJson(string json)
    {
        if (string.IsNullOrWhiteSpace(json))
        {
            throw new ArgumentException("JSON string must be provided.", nameof(json));
        }

        var options = new JsonSerializerOptions
        {
            PropertyNameCaseInsensitive = true,
            ReadCommentHandling = JsonCommentHandling.Skip,
            AllowTrailingCommas = true
        };

        return JsonSerializer.Deserialize<TokenizerConfiguration>(json, options)
            ?? throw new InvalidDataException("Tokenizer configuration could not be deserialized.");
    }
}

/// <summary>
/// A <see cref="JsonConverter{T}"/> for <see cref="long"/> that safely reads JSON numbers
/// stored as floating-point values (e.g. <c>1e30</c> or <c>1000000000000000019884624838656</c>).
/// Values outside the <see cref="long"/> range are clamped to <see cref="long.MaxValue"/>
/// or <see cref="long.MinValue"/> respectively.
/// </summary>
internal sealed class LongFromNumberConverter : JsonConverter<long>
{
    /// <inheritdoc/>
    public override long Read(ref Utf8JsonReader reader, Type typeToConvert, JsonSerializerOptions options)
    {
        if (reader.TokenType == JsonTokenType.Number)
        {
            // Try integer first to preserve exact values for normal token limits (e.g. 4096, 32768).
            if (reader.TryGetInt64(out var intValue))
            {
                return intValue;
            }

            // Fall back to double for large floating-point numbers (e.g. 1e30 for "unlimited").
            if (reader.TryGetDouble(out var doubleValue))
            {
                if (doubleValue >= (double)long.MaxValue)
                {
                    return long.MaxValue;
                }

                if (doubleValue <= (double)long.MinValue)
                {
                    return long.MinValue;
                }

                return (long)doubleValue;
            }
        }

        throw new JsonException($"Cannot convert token '{reader.TokenType}' to {nameof(Int64)}.");
    }

    /// <inheritdoc/>
    public override void Write(Utf8JsonWriter writer, long value, JsonSerializerOptions options)
    {
        writer.WriteNumberValue(value);
    }
}
