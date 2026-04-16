using System.Text.Json.Serialization;

namespace WebExpress.LLM.Model;

/// <summary>
/// Rotary position embedding parameters for the text encoder, which uses different
/// RoPE configurations for full-attention and sliding-attention layers.
/// </summary>
public sealed class TextRopeParameters
{
    /// <summary>
    /// Gets the RoPE configuration applied to full-attention layers.
    /// </summary>
    [JsonPropertyName("full_attention")]
    public RopeEntry FullAttention { get; init; }

    /// <summary>
    /// Gets the RoPE configuration applied to sliding-attention layers.
    /// </summary>
    [JsonPropertyName("sliding_attention")]
    public RopeEntry SlidingAttention { get; init; }
}
