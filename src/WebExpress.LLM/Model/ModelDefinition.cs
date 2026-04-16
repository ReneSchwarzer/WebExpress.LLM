using System;
using WebExpress.LLM.SafeTensors;

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
    /// This is set for non-sharded models that use a single weights file, and null for sharded models.
    /// </summary>
    public ModelWeights Weights { get; init; }

    /// <summary>
    /// Gets the sharded SafeTensor loader used by the model when weights are distributed
    /// across multiple shard files. This is null for non-sharded models.
    /// </summary>
    public ShardedSafeTensorLoader ShardedLoader { get; init; }

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
        ShardedLoader?.Dispose();
        _disposed = true;
    }
}
