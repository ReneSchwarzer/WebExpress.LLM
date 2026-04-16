using System;

namespace WebExpress.LLM.Inference;

/// <summary>
/// Configuration options for text generation during inference.
/// </summary>
public sealed class GenerationConfig
{
    /// <summary>
    /// Gets the maximum number of new tokens that can be generated in a single operation.
    /// </summary>
    public int MaxNewTokens { get; init; } = 32;

    /// <summary>
    /// Gets the temperature value used for sampling randomness in the model's output.
    /// </summary>
    /// <remarks>Higher values increase randomness and creativity in generated results, while lower values
    /// make the output more focused and deterministic. Typical values range from 0.0 to 2.0.</remarks>
    public float Temperature { get; init; } = 1.0f;

    /// <summary>
    /// Gets the maximum number of results to return, or null to return all available results.
    /// </summary>
    public int? TopK { get; init; }

    /// <summary>
    /// Gets the cumulative probability threshold for nucleus sampling (Top-p) used during text generation.
    /// </summary>
    /// <remarks>Set this property to limit the next token selection to the smallest possible set of tokens
    /// whose cumulative probability exceeds the specified value. Lower values make the output more focused and
    /// deterministic, while higher values increase randomness. If null, the default behavior of the underlying model is
    /// used.</remarks>
    public float? TopP { get; init; }

    /// <summary>
    /// Gets the optional seed value used to initialize random number generation.
    /// </summary>
    /// <remarks>If not set, a default seed may be used, resulting in non-deterministic random sequences.
    /// Specify a value to produce repeatable results across runs.</remarks>
    public int? Seed { get; init; }

    /// <summary>
    /// Creates and returns an appropriate sampling strategy based on the configured parameters.
    /// </summary>
    /// <remarks>Only one sampling strategy can be selected at a time. If both TopK and TopP are unset, greedy
    /// sampling is used by default.</remarks>
    /// <returns>An implementation of ISamplingStrategy determined by the current settings. Returns a TopKSampling instance if
    /// TopK is specified, a TopPSampling instance if TopP is specified, or a GreedySampling instance if neither is set.</returns>
    /// <exception cref="InvalidOperationException">Thrown if both TopK and TopP parameters are specified at the same time.</exception>
    public ISamplingStrategy CreateSamplingStrategy()
    {
        if (TopK.HasValue && TopP.HasValue)
        {
            throw new InvalidOperationException("Cannot specify both TopK and TopP sampling.");
        }

        if (TopK.HasValue)
        {
            return new TopKSampling(TopK.Value, Seed);
        }

        if (TopP.HasValue)
        {
            return new TopPSampling(TopP.Value, Seed);
        }

        return new GreedySampling();
    }
}
