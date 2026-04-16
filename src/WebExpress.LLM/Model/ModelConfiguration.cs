namespace WebExpress.LLM.Model;

/// <summary>
/// Configuration parameters for a Gemma-4 transformer model.
/// </summary>
public sealed class ModelConfiguration
{
    /// <summary>
    /// Gets the name of the model.
    /// </summary>
    public string ModelName { get; init; } = string.Empty;

    /// <summary>
    /// Gets the total number of unique tokens in the vocabulary.
    /// </summary>
    public int VocabularySize { get; init; }

    /// <summary>
    /// Gets the maximum number of tokens or characters to include in the context.
    /// </summary>
    public int ContextLength { get; init; }

    /// <summary>
    /// Gets the size of the hidden layer used in the model.
    /// </summary>
    public int HiddenSize { get; init; }

    /// <summary>
    /// Gets the intermediate size value used in processing or calculations.
    /// </summary>
    public int IntermediateSize { get; init; }

    /// <summary>
    /// Gets the number of layers configured for the current instance.
    /// </summary>
    public int NumberOfLayers { get; init; }

    /// <summary>
    /// Gets the number of attention heads used in the model.
    /// </summary>
    /// <remarks>This value determines how many parallel attention mechanisms are applied within each
    /// transformer layer. Increasing the number of attention heads can improve the model's ability to capture different
    /// representation subspaces, but may also increase computational cost.</remarks>
    public int NumberOfAttentionHeads { get; init; }

    /// <summary>
    /// Gets the number of key-value attention heads used in the model.
    /// </summary>
    public int NumberOfKeyValueHeads { get; init; }

    /// <summary>
    /// Gets the epsilon value used to maintain numerical stability during RMS normalization operations.
    /// </summary>
    /// <remarks>This value is typically added to the denominator in RMS normalization to prevent division by
    /// zero or very small numbers. Adjust this value only if you have specific numerical requirements.</remarks>
    public float RmsNormEpsilon { get; init; } = 1e-6f;

    /// <summary>
    /// Gets the current value of the rope angle in degrees.
    /// </summary>
    public float RopeTheta { get; init; } = 10000.0f;

    /// <summary>
    /// Gets the size of each attention head in the model.
    /// </summary>
    public int HeadDimension { get; init; }
}
