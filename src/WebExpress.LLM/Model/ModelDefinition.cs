namespace WebExpress.LLM.Model;

/// <summary>
/// Provides a definition of a model, including its configuration and the associated weight data.
/// </summary>
public sealed class ModelDefinition
{
    /// <summary>
    /// Gets or sets the configuration for the model.
    /// </summary>
    public required ModelConfiguration Configuration { get; init; }

    /// <summary>
    /// Gets the serialized weights used by the model.
    /// </summary>
    public required byte[] Weights { get; init; }
}
