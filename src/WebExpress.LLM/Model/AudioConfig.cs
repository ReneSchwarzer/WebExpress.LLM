using System.Collections.Generic;
using System.Text.Json.Serialization;

namespace WebExpress.LLM.Model;

/// <summary>
/// Configuration parameters for the audio encoder component of a Gemma-4 model,
/// corresponding to the <c>audio_config</c> section of a HuggingFace <c>config.json</c>.
/// </summary>
public sealed class AudioConfig
{
    /// <summary>
    /// Gets the model type identifier (e.g. "gemma4_audio").
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
    /// Gets the output projection dimensionality.
    /// </summary>
    [JsonPropertyName("output_proj_dims")]
    public int OutputProjDimensions { get; init; }

    /// <summary>
    /// Gets the epsilon value used in RMS normalisation layers.
    /// </summary>
    [JsonPropertyName("rms_norm_eps")]
    public float RmsNormEpsilon { get; init; } = 1e-6f;

    /// <summary>
    /// Gets the convolution kernel size used in the audio feature extractor.
    /// </summary>
    [JsonPropertyName("conv_kernel_size")]
    public int ConvKernelSize { get; init; }

    /// <summary>
    /// Gets the number of channels for each subsampling convolution stage.
    /// </summary>
    [JsonPropertyName("subsampling_conv_channels")]
    public IReadOnlyList<int> SubsamplingConvChannels { get; init; } = [];

    /// <summary>
    /// Gets the activation function used in hidden layers (e.g. "silu").
    /// </summary>
    [JsonPropertyName("hidden_act")]
    public string HiddenActivation { get; init; } = string.Empty;

    /// <summary>
    /// Gets the residual connection weight applied within audio encoder layers.
    /// </summary>
    [JsonPropertyName("residual_weight")]
    public float ResidualWeight { get; init; }

    /// <summary>
    /// Gets the logit cap value applied to attention scores.
    /// </summary>
    [JsonPropertyName("attention_logit_cap")]
    public float AttentionLogitCap { get; init; }

    /// <summary>
    /// Gets the chunk size used in chunked attention.
    /// </summary>
    [JsonPropertyName("attention_chunk_size")]
    public int AttentionChunkSize { get; init; }

    /// <summary>
    /// Gets the left context size (number of preceding frames) visible to each attention chunk.
    /// </summary>
    [JsonPropertyName("attention_context_left")]
    public int AttentionContextLeft { get; init; }

    /// <summary>
    /// Gets the right context size (number of following frames) visible to each attention chunk.
    /// </summary>
    [JsonPropertyName("attention_context_right")]
    public int AttentionContextRight { get; init; }

    /// <summary>
    /// Gets the value used to mask invalid attention logit positions.
    /// </summary>
    [JsonPropertyName("attention_invalid_logits_value")]
    public float AttentionInvalidLogitsValue { get; init; }

    /// <summary>
    /// Gets a value indicating whether clipped linear activations are used.
    /// </summary>
    [JsonPropertyName("use_clipped_linears")]
    public bool UseClippedLinears { get; init; }

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
