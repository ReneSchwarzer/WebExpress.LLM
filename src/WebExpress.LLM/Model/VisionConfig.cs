using System.Text.Json.Serialization;

namespace WebExpress.LLM.Model;

/// <summary>
/// Configuration parameters for the vision encoder component of a Gemma-4 model,
/// corresponding to the <c>vision_config</c> section of a HuggingFace <c>config.json</c>.
/// </summary>
public sealed class VisionConfig
{
    /// <summary>
    /// Gets the model type identifier (e.g. "gemma4_vision").
    /// </summary>
    [JsonPropertyName("model_type")]
    public string ModelType { get; init; } = string.Empty;

    /// <summary>
    /// Gets the dimensionality of the hidden states.
    /// </summary>
    [JsonPropertyName("hidden_size")]
    public int HiddenSize { get; init; }

    /// <summary>
    /// Gets the number of transformer layers.
    /// </summary>
    [JsonPropertyName("num_hidden_layers")]
    public int NumberOfLayers { get; init; }

    /// <summary>
    /// Gets the number of attention heads per layer.
    /// </summary>
    [JsonPropertyName("num_attention_heads")]
    public int NumberOfAttentionHeads { get; init; }

    /// <summary>
    /// Gets the number of key-value attention heads per layer.
    /// </summary>
    [JsonPropertyName("num_key_value_heads")]
    public int NumberOfKeyValueHeads { get; init; }

    /// <summary>
    /// Gets the size of the feed-forward intermediate layer.
    /// </summary>
    [JsonPropertyName("intermediate_size")]
    public int IntermediateSize { get; init; }

    /// <summary>
    /// Gets the per-head dimensionality for attention.
    /// </summary>
    [JsonPropertyName("head_dim")]
    public int HeadDimension { get; init; }

    /// <summary>
    /// Gets the per-head dimensionality for global attention.
    /// </summary>
    [JsonPropertyName("global_head_dim")]
    public int GlobalHeadDimension { get; init; }

    /// <summary>
    /// Gets the maximum number of position embeddings supported.
    /// </summary>
    [JsonPropertyName("max_position_embeddings")]
    public int MaxPositionEmbeddings { get; init; }

    /// <summary>
    /// Gets the size of the position embedding table.
    /// </summary>
    [JsonPropertyName("position_embedding_size")]
    public int PositionEmbeddingSize { get; init; }

    /// <summary>
    /// Gets the patch size (height and width in pixels) used for image tokenisation.
    /// </summary>
    [JsonPropertyName("patch_size")]
    public int PatchSize { get; init; }

    /// <summary>
    /// Gets the kernel size used for pooling vision tokens.
    /// </summary>
    [JsonPropertyName("pooling_kernel_size")]
    public int PoolingKernelSize { get; init; }

    /// <summary>
    /// Gets the default number of soft tokens produced per image.
    /// </summary>
    [JsonPropertyName("default_output_length")]
    public int DefaultOutputLength { get; init; }

    /// <summary>
    /// Gets the epsilon value used in RMS normalisation layers.
    /// </summary>
    [JsonPropertyName("rms_norm_eps")]
    public float RmsNormEpsilon { get; init; } = 1e-6f;

    /// <summary>
    /// Gets the activation function used in hidden layers (e.g. "gelu_pytorch_tanh").
    /// </summary>
    [JsonPropertyName("hidden_activation")]
    public string HiddenActivation { get; init; } = string.Empty;

    /// <summary>
    /// Gets the rotary position embedding parameters for the vision encoder.
    /// </summary>
    [JsonPropertyName("rope_parameters")]
    public RopeEntry RopeParameters { get; init; }

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
    /// Gets a value indicating whether clipped linear activations are used.
    /// </summary>
    [JsonPropertyName("use_clipped_linears")]
    public bool UseClippedLinears { get; init; }

    /// <summary>
    /// Gets a value indicating whether input features are standardised before processing.
    /// </summary>
    [JsonPropertyName("standardize")]
    public bool Standardize { get; init; }

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
}
