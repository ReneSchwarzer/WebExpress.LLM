using System;
using System.Collections.Generic;
using System.Linq;

namespace WebExpress.LLM.Inference;

/// <summary>
/// Implements top-k sampling by selecting from the k tokens with the highest probabilities.
/// </summary>
public sealed class TopKSampling : ISamplingStrategy
{
    private readonly int _k;
    private readonly Random _random;

    public TopKSampling(int k, int? seed = null)
    {
        if (k <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(k), "k must be greater than zero.");
        }

        _k = k;
        _random = seed.HasValue ? new Random(seed.Value) : new Random();
    }

    public int Sample(IReadOnlyList<float> logits)
    {
        ArgumentNullException.ThrowIfNull(logits);

        if (logits.Count == 0)
        {
            throw new ArgumentException("Logits must not be empty.", nameof(logits));
        }

        var topK = logits
            .Select((logit, index) => (logit, index))
            .OrderByDescending(item => item.logit)
            .Take(Math.Min(_k, logits.Count))
            .ToArray();

        var probabilities = Softmax(topK.Select(item => item.logit).ToArray());
        return topK[SampleFromDistribution(probabilities)].index;
    }

    private static float[] Softmax(float[] logits)
    {
        var maxLogit = logits.Max();
        var expSum = logits.Sum(logit => MathF.Exp(logit - maxLogit));
        return logits.Select(logit => MathF.Exp(logit - maxLogit) / expSum).ToArray();
    }

    private int SampleFromDistribution(float[] probabilities)
    {
        var sample = _random.NextSingle();
        var cumulative = 0.0f;

        for (var i = 0; i < probabilities.Length; i++)
        {
            cumulative += probabilities[i];
            if (sample < cumulative)
            {
                return i;
            }
        }

        return probabilities.Length - 1;
    }
}
