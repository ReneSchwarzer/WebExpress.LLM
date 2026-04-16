using System;
using System.Collections.Generic;
using System.Linq;

namespace WebExpress.LLM.Inference;

/// <summary>
/// Implements nucleus (top-p) sampling by selecting from the smallest set of tokens whose cumulative
/// probability exceeds the threshold p.
/// </summary>
public sealed class TopPSampling : ISamplingStrategy
{
    private readonly float _p;
    private readonly Random _random;

    public TopPSampling(float p, int? seed = null)
    {
        if (p <= 0.0f || p > 1.0f)
        {
            throw new ArgumentOutOfRangeException(nameof(p), "p must be in the range (0, 1].");
        }

        _p = p;
        _random = seed.HasValue ? new Random(seed.Value) : new Random();
    }

    public int Sample(IReadOnlyList<float> logits)
    {
        ArgumentNullException.ThrowIfNull(logits);

        if (logits.Count == 0)
        {
            throw new ArgumentException("Logits must not be empty.", nameof(logits));
        }

        var sortedIndices = logits
            .Select((logit, index) => (logit, index))
            .OrderByDescending(item => item.logit)
            .ToArray();

        var probabilities = Softmax(sortedIndices.Select(item => item.logit).ToArray());

        var cumulativeProbability = 0.0f;
        var nucleusSize = 0;

        for (var i = 0; i < probabilities.Length; i++)
        {
            cumulativeProbability += probabilities[i];
            nucleusSize++;

            if (cumulativeProbability >= _p)
            {
                break;
            }
        }

        var nucleus = sortedIndices.Take(nucleusSize).ToArray();
        var nucleusProbabilities = probabilities.Take(nucleusSize).ToArray();

        var normalizedProbabilities = NormalizeProbabilities(nucleusProbabilities);
        var selectedIndex = SampleFromDistribution(normalizedProbabilities);

        return nucleus[selectedIndex].index;
    }

    private static float[] Softmax(float[] logits)
    {
        var maxLogit = logits.Max();
        var expSum = logits.Sum(logit => MathF.Exp(logit - maxLogit));
        return logits.Select(logit => MathF.Exp(logit - maxLogit) / expSum).ToArray();
    }

    private static float[] NormalizeProbabilities(float[] probabilities)
    {
        var sum = probabilities.Sum();
        return probabilities.Select(p => p / sum).ToArray();
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
