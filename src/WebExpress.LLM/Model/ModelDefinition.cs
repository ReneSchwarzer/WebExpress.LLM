using System;

namespace WebExpress.LLM.Model;

/// <summary>
/// Provides a definition of a model, including its configuration and the associated weight data.
/// </summary>
public sealed class ModelDefinition : IDisposable
{
    private bool _disposed;

    /// <summary>
    /// Gets or sets the configuration for the model.
    /// </summary>
    public required ModelConfiguration Configuration { get; init; }

    /// <summary>
    /// Gets the serialized weights used by the model.
    /// </summary>
    public required ModelWeights Weights { get; init; }

    /// <summary>
    /// Disposes the ModelDefinition and releases associated resources.
    /// </summary>
    public void Dispose()
    {
        if (_disposed)
        {
            return;
        }

        Weights?.Dispose();
        _disposed = true;
    }
}
