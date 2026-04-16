using System.Collections.Generic;
using System.Text.Json.Serialization;

namespace WebExpress.LLM.Model;

/// <summary>
/// Configuration parameters for the text encoder component of a Gemma-4 model,
/// corresponding to the <c>text_config</c> section of a HuggingFace <c>config.json</c>.
/// </summary>
public sealed class TextConfig
{
    /// <summary>
    /// Gets the model type identifier (e.g. "gemma4_text").
    /// </summary>
    [JsonPropertyName("model_type")]
    public string ModelType { get; init; } = string.Empty;

    /// <summary>
    /// Gets the total number of unique tokens in the vocabulary.
    /// </summary>
    [JsonPropertyName("vocab_size")]
    public int VocabularySize { get; init; }

    /// <summary>
    /// Gets the vocabulary size used for per-layer input projections.
    /// </summary>
    [JsonPropertyName("vocab_size_per_layer_input")]
    public int VocabSizePerLayerInput { get; init; }

    /// <summary>
    /// Gets the maximum sequence length (context window) supported by the model.
    /// </summary>
    [JsonPropertyName("max_position_embeddings")]
    public int MaxPositionEmbeddings { get; init; }

    /// <summary>
    /// Gets the dimensionality of the hidden states.
    /// </summary>
    [JsonPropertyName("hidden_size")]
    public int HiddenSize { get; init; }

    /// <summary>
    /// Gets the hidden size used for per-layer input projections.
    /// </summary>
    [JsonPropertyName("hidden_size_per_layer_input")]
    public int HiddenSizePerLayerInput { get; init; }

    /// <summary>
    /// Gets the size of the feed-forward intermediate layer.
    /// </summary>
    [JsonPropertyName("intermediate_size")]
    public int IntermediateSize { get; init; }

    /// <summary>
    /// Gets the number of transformer layers.
    /// </summary>
    [JsonPropertyName("num_hidden_layers")]
    public int NumberOfLayers { get; init; }

    /// <summary>
    /// Gets the number of query attention heads per layer.
    /// </summary>
    [JsonPropertyName("num_attention_heads")]
    public int NumberOfAttentionHeads { get; init; }

    /// <summary>
    /// Gets the number of key-value attention heads per layer.
    /// </summary>
    [JsonPropertyName("num_key_value_heads")]
    public int NumberOfKeyValueHeads { get; init; }

    /// <summary>
    /// Gets the per-head dimensionality for sliding-window (local) attention.
    /// </summary>
    [JsonPropertyName("head_dim")]
    public int HeadDimension { get; init; }

    /// <summary>
    /// Gets the per-head dimensionality for full (global) attention.
    /// </summary>
    [JsonPropertyName("global_head_dim")]
    public int GlobalHeadDimension { get; init; }

    /// <summary>
    /// Gets the epsilon value used in RMS normalisation layers.
    /// </summary>
    [JsonPropertyName("rms_norm_eps")]
    public float RmsNormEpsilon { get; init; } = 1e-6f;

    /// <summary>
    /// Gets the rotary position embedding parameters, keyed by attention type.
    /// </summary>
    [JsonPropertyName("rope_parameters")]
    public TextRopeParameters RopeParameters { get; init; }

    /// <summary>
    /// Gets the sliding-window size for local attention layers.
    /// </summary>
    [JsonPropertyName("sliding_window")]
    public int SlidingWindow { get; init; }

    /// <summary>
    /// Gets the activation function used in hidden layers.
    /// </summary>
    [JsonPropertyName("hidden_activation")]
    public string HiddenActivation { get; init; } = string.Empty;

    /// <summary>
    /// Gets the soft-capping value applied to the final logits.
    /// </summary>
    [JsonPropertyName("final_logit_softcapping")]
    public float FinalLogitSoftcapping { get; init; }

    /// <summary>
    /// Gets a value indicating whether attention bias terms are used.
    /// </summary>
    [JsonPropertyName("attention_bias")]
    public bool AttentionBias { get; init; }

    /// <summary>
    /// Gets the dropout probability applied to attention weights.
    /// </summary>
    [JsonPropertyName("attention_dropout")]
    public float AttentionDropout { get; init; }

    /// <summary>
    /// Gets a value indicating whether key and value projections share the same weight.
    /// </summary>
    [JsonPropertyName("attention_k_eq_v")]
    public bool AttentionKeyEqualsValue { get; init; }

    /// <summary>
    /// Gets the beginning-of-sequence token identifier.
    /// </summary>
    [JsonPropertyName("bos_token_id")]
    public int BosTokenId { get; init; }

    /// <summary>
    /// Gets the end-of-sequence token identifier.
    /// </summary>
    [JsonPropertyName("eos_token_id")]
    [JsonConverter(typeof(IntOrArrayConverter))]
    public int EosTokenId { get; init; }

    /// <summary>
    /// Gets the padding token identifier.
    /// </summary>
    [JsonPropertyName("pad_token_id")]
    public int PadTokenId { get; init; }

    /// <summary>
    /// Gets the data type used during model execution (e.g. "bfloat16").
    /// </summary>
    [JsonPropertyName("dtype")]
    public string Dtype { get; init; } = string.Empty;

    /// <summary>
    /// Gets a value indicating whether word embeddings are shared between input and output projections.
    /// </summary>
    [JsonPropertyName("tie_word_embeddings")]
    public bool TieWordEmbeddings { get; init; }

    /// <summary>
    /// Gets a value indicating whether the key-value cache is used during inference.
    /// </summary>
    [JsonPropertyName("use_cache")]
    public bool UseCache { get; init; }

    /// <summary>
    /// Gets a value indicating whether a double-wide MLP variant is used.
    /// </summary>
    [JsonPropertyName("use_double_wide_mlp")]
    public bool UseDoubleWideMlp { get; init; }

    /// <summary>
    /// Gets a value indicating whether mixture-of-experts blocks are enabled.
    /// </summary>
    [JsonPropertyName("enable_moe_block")]
    public bool EnableMoeBlock { get; init; }

    /// <summary>
    /// Gets the list of attention layer types (e.g. "sliding_attention", "full_attention").
    /// </summary>
    [JsonPropertyName("layer_types")]
    public IReadOnlyList<string> LayerTypes { get; init; } = [];

    /// <summary>
    /// Gets the number of layers that share the same key-value cache.
    /// </summary>
    [JsonPropertyName("num_kv_shared_layers")]
    public int NumberOfKvSharedLayers { get; init; }
}
