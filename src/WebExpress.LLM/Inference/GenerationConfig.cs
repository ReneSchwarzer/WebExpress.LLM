using System;
using System.Collections.Generic;

namespace WebExpress.LLM.Inference;

/// <summary>
/// Configuration options for text generation during inference.
/// </summary>
public sealed class GenerationConfig
{
    public int MaxNewTokens { get; init; } = 32;

    public float Temperature { get; init; } = 1.0f;

    public int? TopK { get; init; }

    public float? TopP { get; init; }

    public int? Seed { get; init; }

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
