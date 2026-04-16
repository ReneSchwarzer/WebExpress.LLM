namespace WebExpress.LLM.Model;

public sealed class ModelDefinition
{
    public required ModelConfiguration Configuration { get; init; }

    public required byte[] Weights { get; init; }
}
