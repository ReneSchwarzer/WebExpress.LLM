namespace WebExpress.LLM.Model;

/// <summary>
/// Configuration parameters for a Gemma-4 transformer model.
/// </summary>
public sealed class ModelConfiguration
{
    public string ModelName { get; init; } = string.Empty;

    public int VocabularySize { get; init; }

    public int ContextLength { get; init; }

    public int HiddenSize { get; init; }

    public int IntermediateSize { get; init; }

    public int NumberOfLayers { get; init; }

    public int NumberOfAttentionHeads { get; init; }

    public int NumberOfKeyValueHeads { get; init; }

    public float RmsNormEpsilon { get; init; } = 1e-6f;

    public float RopeTheta { get; init; } = 10000.0f;

    public int HeadDimension { get; init; }
}
