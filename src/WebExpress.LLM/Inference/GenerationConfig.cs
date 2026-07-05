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
    /// Gets the repetition penalty factor applied to already-generated tokens.
    /// Values greater than 1.0 penalize repeats; 1.0 disables the penalty.
    /// </summary>
    public float RepetitionPenalty { get; init; } = 1.1f;

    /// <summary>
    /// Creates and returns an appropriate sampling strategy based on the configured parameters.
    /// </summary>
    /// <remarks>
    /// The strategy is selected as follows:
    /// <list type="bullet">
    ///   <item>No sampling controls (no <see cref="TopK"/>, no <see cref="TopP"/>, and a default
    ///     <see cref="Temperature"/> of 1.0) — <see cref="GreedySampling"/>.</item>
    ///   <item>Only <see cref="TopK"/> with a default temperature — <see cref="TopKSampling"/>.</item>
    ///   <item>Only <see cref="TopP"/> with a default temperature — <see cref="TopPSampling"/>.</item>
    ///   <item>Any other combination — including both <see cref="TopK"/> and <see cref="TopP"/>
    ///     together, or a non-default <see cref="Temperature"/> — <see cref="CombinedSampling"/>,
    ///     which applies the repetition penalty, temperature, top-k, and top-p filters in sequence.</item>
    /// </list>
    /// </remarks>
    /// <returns>An implementation of <see cref="ISamplingStrategy"/> determined by the current settings.</returns>
    public ISamplingStrategy CreateSamplingStrategy()
    {
        // A temperature other than 1.0 means the distribution must be scaled, which only the combined
        // pipeline honors; top-k and top-p can also be applied together there.
        var hasTemperature = Temperature != 1.0f;

        if (!hasTemperature && TopK.HasValue && !TopP.HasValue)
        {
            return new TopKSampling(TopK.Value, Seed, RepetitionPenalty);
        }

        if (!hasTemperature && TopP.HasValue && !TopK.HasValue)
        {
            return new TopPSampling(TopP.Value, Seed, RepetitionPenalty);
        }

        if (!hasTemperature && !TopK.HasValue && !TopP.HasValue)
        {
            return new GreedySampling(RepetitionPenalty);
        }

        return new CombinedSampling(Temperature, TopK, TopP, Seed, RepetitionPenalty);
    }
}
