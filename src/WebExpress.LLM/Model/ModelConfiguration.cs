namespace WebExpress.LLM.Model;

public sealed class ModelConfiguration
{
    public string ModelName { get; init; } = string.Empty;

    public int VocabularySize { get; init; }

    public int ContextLength { get; init; }
}
