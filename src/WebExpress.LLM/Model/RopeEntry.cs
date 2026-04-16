using System.Text.Json.Serialization;

namespace WebExpress.LLM.Model;

/// <summary>
/// Rotary position embedding (RoPE) parameters for a single attention type.
/// </summary>
public sealed class RopeEntry
{
    /// <summary>
    /// Gets the base frequency used for the rotary position embedding.
    /// </summary>
    [JsonPropertyName("rope_theta")]
    public float RopeTheta { get; init; } = 10000.0f;

    /// <summary>
    /// Gets the RoPE scaling strategy (e.g. "default", "proportional").
    /// </summary>
    [JsonPropertyName("rope_type")]
    public string RopeType { get; init; } = "default";

    /// <summary>
    /// Gets the fraction of the head dimension that is rotated (used by "proportional" type).
    /// </summary>
    [JsonPropertyName("partial_rotary_factor")]
    public float PartialRotaryFactor { get; init; } = 1.0f;
}
