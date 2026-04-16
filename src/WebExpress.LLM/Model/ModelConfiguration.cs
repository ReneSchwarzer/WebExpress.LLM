using System.Collections.Generic;
using System.Text.Json.Serialization;

namespace WebExpress.LLM.Model;

/// <summary>
/// Configuration parameters for a Gemma-4 transformer model.
/// </summary>
/// <remarks>
/// Supports both flat HuggingFace config formats (where properties such as <c>vocab_size</c>
/// appear at the top level) and the nested Gemma-4 multi-modal format (where the same
/// properties live inside a <c>text_config</c> sub-object).  When a flat property is absent
/// (i.e. its value is 0 / null / empty), the corresponding value is resolved from
/// <see cref="TextConfig"/> automatically, so callers always receive a populated value.
/// </remarks>
public sealed class ModelConfiguration
{
    // -------------------------------------------------------------------------
    // Backing fields for properties that fall back to TextConfig when absent
    // -------------------------------------------------------------------------

    private readonly int _vocabularySize;
    private readonly int _contextLength;
    private readonly int _hiddenSize;
    private readonly int _intermediateSize;
    private readonly int _numberOfLayers;
    private readonly int _numberOfAttentionHeads;
    private readonly int _numberOfKeyValueHeads;
    private readonly float _rmsNormEpsilon;
    private readonly float _ropeTheta;
    private readonly int _headDimension;

    // -------------------------------------------------------------------------
    // Top-level identity / meta properties
    // -------------------------------------------------------------------------

    /// <summary>
    /// Gets the name of the model.
    /// </summary>
    [JsonPropertyName("model_name")]
    public string ModelName { get; init; } = string.Empty;

    /// <summary>
    /// Gets the model type identifier (e.g. "gemma4").
    /// </summary>
    [JsonPropertyName("model_type")]
    public string ModelType { get; init; } = string.Empty;

    /// <summary>
    /// Gets the list of architecture class names declared in the configuration.
    /// </summary>
    [JsonPropertyName("architectures")]
    public IReadOnlyList<string> Architectures { get; init; } = [];

    /// <summary>
    /// Gets the data type used during model execution (e.g. "bfloat16").
    /// </summary>
    [JsonPropertyName("dtype")]
    public string Dtype { get; init; } = string.Empty;

    /// <summary>
    /// Gets the initializer range for weight initialisation.
    /// </summary>
    [JsonPropertyName("initializer_range")]
    public float InitializerRange { get; init; } = 0.02f;

    /// <summary>
    /// Gets a value indicating whether word embeddings are shared between input and output projections.
    /// </summary>
    [JsonPropertyName("tie_word_embeddings")]
    public bool TieWordEmbeddings { get; init; }

    /// <summary>
    /// Gets the transformers library version that produced this configuration.
    /// </summary>
    [JsonPropertyName("transformers_version")]
    public string TransformersVersion { get; init; } = string.Empty;

    // -------------------------------------------------------------------------
    // Multi-modal token identifiers
    // -------------------------------------------------------------------------

    /// <summary>
    /// Gets the token identifier used to represent audio content.
    /// </summary>
    [JsonPropertyName("audio_token_id")]
    public int AudioTokenId { get; init; }

    /// <summary>
    /// Gets the beginning-of-audio token identifier.
    /// </summary>
    [JsonPropertyName("boa_token_id")]
    public int BoaTokenId { get; init; }

    /// <summary>
    /// Gets the beginning-of-image token identifier.
    /// </summary>
    [JsonPropertyName("boi_token_id")]
    public int BoiTokenId { get; init; }

    /// <summary>
    /// Gets the end-of-audio token identifier.
    /// </summary>
    [JsonPropertyName("eoa_token_id")]
    public int EoaTokenId { get; init; }

    /// <summary>
    /// Gets the end-of-audio token index used during generation.
    /// </summary>
    [JsonPropertyName("eoa_token_index")]
    public int EoaTokenIndex { get; init; }

    /// <summary>
    /// Gets the end-of-image token identifier.
    /// </summary>
    [JsonPropertyName("eoi_token_id")]
    public int EoiTokenId { get; init; }

    /// <summary>
    /// Gets the end-of-sequence token identifier (or the first value when the field is an array).
    /// </summary>
    [JsonPropertyName("eos_token_id")]
    public int EosTokenId { get; init; }

    /// <summary>
    /// Gets the token identifier used to represent image content.
    /// </summary>
    [JsonPropertyName("image_token_id")]
    public int ImageTokenId { get; init; }

    /// <summary>
    /// Gets the token identifier used to represent video content.
    /// </summary>
    [JsonPropertyName("video_token_id")]
    public int VideoTokenId { get; init; }

    /// <summary>
    /// Gets the number of soft vision tokens generated per image.
    /// </summary>
    [JsonPropertyName("vision_soft_tokens_per_image")]
    public int VisionSoftTokensPerImage { get; init; }

    // -------------------------------------------------------------------------
    // Nested sub-model configurations
    // -------------------------------------------------------------------------

    /// <summary>
    /// Gets the configuration for the text encoder component.
    /// </summary>
    [JsonPropertyName("text_config")]
    public TextConfig TextConfig { get; init; }

    /// <summary>
    /// Gets the configuration for the audio encoder component.
    /// </summary>
    [JsonPropertyName("audio_config")]
    public AudioConfig AudioConfig { get; init; }

    /// <summary>
    /// Gets the configuration for the vision encoder component.
    /// </summary>
    [JsonPropertyName("vision_config")]
    public VisionConfig VisionConfig { get; init; }

    // -------------------------------------------------------------------------
    // Flat text-model properties with TextConfig fallback
    //
    // Each property reads from its backing field when the caller supplied an
    // explicit value (> 0 / non-default), and falls back to the equivalent
    // TextConfig property when the flat JSON field was absent (Gemma-4 nested
    // format).
    // -------------------------------------------------------------------------

    /// <summary>
    /// Gets the total number of unique tokens in the vocabulary.
    /// </summary>
    /// <remarks>
    /// When absent at the top level, this value is resolved from <see cref="TextConfig"/>.
    /// </remarks>
    [JsonPropertyName("vocab_size")]
    public int VocabularySize
    {
        get => _vocabularySize > 0 ? _vocabularySize : TextConfig?.VocabularySize ?? 0;
        init => _vocabularySize = value;
    }

    /// <summary>
    /// Gets the maximum number of tokens in the context window.
    /// </summary>
    /// <remarks>
    /// When absent at the top level, this value is resolved from <see cref="TextConfig"/>.
    /// </remarks>
    [JsonPropertyName("max_position_embeddings")]
    public int ContextLength
    {
        get => _contextLength > 0 ? _contextLength : TextConfig?.MaxPositionEmbeddings ?? 0;
        init => _contextLength = value;
    }

    /// <summary>
    /// Gets the dimensionality of the hidden states.
    /// </summary>
    /// <remarks>
    /// When absent at the top level, this value is resolved from <see cref="TextConfig"/>.
    /// </remarks>
    [JsonPropertyName("hidden_size")]
    public int HiddenSize
    {
        get => _hiddenSize > 0 ? _hiddenSize : TextConfig?.HiddenSize ?? 0;
        init => _hiddenSize = value;
    }

    /// <summary>
    /// Gets the size of the feed-forward intermediate layer.
    /// </summary>
    /// <remarks>
    /// When absent at the top level, this value is resolved from <see cref="TextConfig"/>.
    /// </remarks>
    [JsonPropertyName("intermediate_size")]
    public int IntermediateSize
    {
        get => _intermediateSize > 0 ? _intermediateSize : TextConfig?.IntermediateSize ?? 0;
        init => _intermediateSize = value;
    }

    /// <summary>
    /// Gets the number of transformer layers.
    /// </summary>
    /// <remarks>
    /// When absent at the top level, this value is resolved from <see cref="TextConfig"/>.
    /// </remarks>
    [JsonPropertyName("num_hidden_layers")]
    public int NumberOfLayers
    {
        get => _numberOfLayers > 0 ? _numberOfLayers : TextConfig?.NumberOfLayers ?? 0;
        init => _numberOfLayers = value;
    }

    /// <summary>
    /// Gets the number of attention heads used in the model.
    /// </summary>
    /// <remarks>This value determines how many parallel attention mechanisms are applied within each
    /// transformer layer. Increasing the number of attention heads can improve the model's ability to capture different
    /// representation subspaces, but may also increase computational cost.
    /// When absent at the top level, this value is resolved from <see cref="TextConfig"/>.</remarks>
    [JsonPropertyName("num_attention_heads")]
    public int NumberOfAttentionHeads
    {
        get => _numberOfAttentionHeads > 0 ? _numberOfAttentionHeads : TextConfig?.NumberOfAttentionHeads ?? 0;
        init => _numberOfAttentionHeads = value;
    }

    /// <summary>
    /// Gets the number of key-value attention heads used in the model.
    /// </summary>
    /// <remarks>
    /// When absent at the top level, this value is resolved from <see cref="TextConfig"/>.
    /// </remarks>
    [JsonPropertyName("num_key_value_heads")]
    public int NumberOfKeyValueHeads
    {
        get => _numberOfKeyValueHeads > 0 ? _numberOfKeyValueHeads : TextConfig?.NumberOfKeyValueHeads ?? 0;
        init => _numberOfKeyValueHeads = value;
    }

    /// <summary>
    /// Gets the epsilon value used to maintain numerical stability during RMS normalization operations.
    /// </summary>
    /// <remarks>This value is typically added to the denominator in RMS normalization to prevent division by
    /// zero or very small numbers. Adjust this value only if you have specific numerical requirements.
    /// When absent at the top level, this value is resolved from <see cref="TextConfig"/>.</remarks>
    [JsonPropertyName("rms_norm_eps")]
    public float RmsNormEpsilon
    {
        get => _rmsNormEpsilon != 0f ? _rmsNormEpsilon : TextConfig?.RmsNormEpsilon ?? 1e-6f;
        init => _rmsNormEpsilon = value;
    }

    /// <summary>
    /// Gets the base frequency used for rotary position embeddings.
    /// </summary>
    /// <remarks>
    /// When absent at the top level, this value is resolved from the sliding-attention entry in
    /// <see cref="TextConfig"/>.<see cref="TextConfig.RopeParameters"/>.
    /// </remarks>
    [JsonPropertyName("rope_theta")]
    public float RopeTheta
    {
        get
        {
            if (_ropeTheta != 0f)
            {
                return _ropeTheta;
            }

            return TextConfig?.RopeParameters?.SlidingAttention?.RopeTheta
                ?? TextConfig?.RopeParameters?.FullAttention?.RopeTheta
                ?? 10000.0f;
        }
        init => _ropeTheta = value;
    }

    /// <summary>
    /// Gets the size of each attention head in the model.
    /// </summary>
    /// <remarks>
    /// When absent at the top level, this value is resolved from <see cref="TextConfig"/>.
    /// </remarks>
    [JsonPropertyName("head_dim")]
    public int HeadDimension
    {
        get => _headDimension > 0 ? _headDimension : TextConfig?.HeadDimension ?? 0;
        init => _headDimension = value;
    }
}
