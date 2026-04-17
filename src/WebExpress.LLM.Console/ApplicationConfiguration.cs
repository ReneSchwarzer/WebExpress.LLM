namespace WebExpress.LLM.Console;

/// <summary>
/// Represents the application configuration loaded from the XML configuration file.
/// Contains all parameters required for operation including model paths, inference settings,
/// and runtime behavior options.
/// </summary>
public sealed class ApplicationConfiguration
{
    /// <summary>
    /// Gets the name of the model to use.
    /// </summary>
    public string ModelName { get; init; } = string.Empty;

    /// <summary>
    /// Gets the path to the model directory containing configuration and weights files.
    /// </summary>
    public string ModelPath { get; init; } = string.Empty;

    /// <summary>
    /// Gets the maximum number of new tokens to generate during inference.
    /// </summary>
    public int MaxNewTokens { get; init; } = 100;

    /// <summary>
    /// Gets the temperature value for sampling randomness.
    /// </summary>
    /// <remarks>Higher values increase randomness, lower values make output more deterministic.</remarks>
    public float Temperature { get; init; } = 1.0f;

    /// <summary>
    /// Gets the top-k value for sampling, or null if top-k sampling is not used.
    /// </summary>
    public int? TopK { get; init; }

    /// <summary>
    /// Gets the top-p (nucleus) value for sampling, or null if top-p sampling is not used.
    /// </summary>
    public float? TopP { get; init; }

    /// <summary>
    /// Gets the seed value for random number generation, or null for non-deterministic behavior.
    /// </summary>
    public int? Seed { get; init; }

    /// <summary>
    /// Gets the tokenizer type to use (e.g., "byte", "sentencepiece").
    /// </summary>
    public string TokenizerType { get; init; } = "byte";

    /// <summary>
    /// Gets the path to the SentencePiece .model file, relative to the model directory.
    /// Only used when <see cref="TokenizerType"/> is "sentencepiece".
    /// Defaults to "tokenizer.json" when not specified.
    /// </summary>
    public string TokenizerModelPath { get; init; } = "tokenizer.json";

    /// <summary>
    /// Gets a value indicating whether to use deterministic inference engine for testing.
    /// </summary>
    public bool UseDeterministicEngine { get; init; } = false;
}
