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

    /// <summary>
    /// Initializes a new instance of the TopPSampling class with the specified threshold  
    /// for cumulative probability.
    /// </summary>
    /// <param name="p">
    /// The threshold for cumulative probability. Must be greater than 0 and less than or equal to 1.
    /// </param>
    /// <param name="seed">
    /// An optional seed value for the random number generator.  
    /// If not provided, a random seed is used.
    /// </param>
    /// <exception cref="ArgumentOutOfRangeException">
    /// Thrown when <paramref name="p"/> is less than or equal to 0 or greater than 1.
    /// </exception>
    public TopPSampling(float p, int? seed = null)
    {
        if (p <= 0.0f || p > 1.0f)
        {
            throw new ArgumentOutOfRangeException(nameof(p), "p must be in the range (0, 1].");
        }

        _p = p;
        _random = seed.HasValue ? new Random(seed.Value) : new Random();
    }

    /// <summary>
    /// Samples an index from the input logits using nucleus (top-p) sampling.
    /// </summary>
    /// <remarks>Nucleus sampling selects the smallest set of logits whose cumulative probability exceeds a
    /// predefined threshold. This method is commonly used in probabilistic text generation and similar
    /// applications.</remarks>
    /// <param name="logits">A read-only list of logit values representing unnormalized log probabilities. Cannot be null or empty.</param>
    /// <returns>The index of the selected logit after applying nucleus sampling.</returns>
    /// <exception cref="ArgumentException">Thrown if logits is empty.</exception>
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

    /// <summary>
    /// Computes the softmax probability distribution for the specified array of logits.
    /// </summary>
    /// <remarks>
    /// The softmax function is commonly used in machine learning applications to convert raw values (logits)  
    /// into a probability distribution. This implementation is numerically stable even for large logit values.
    /// </remarks>
    /// <param name="logits">
    /// The array of logits for which the softmax values are to be computed. Must not be null and must contain  
    /// at least one element.
    /// </param>
    /// <returns>
    /// An array of floating‑point numbers representing the softmax probabilities for each element in  
    /// <paramref name="logits"/>. The sum of all values is 1.
    /// </returns>
    private static float[] Softmax(float[] logits)
    {
        var maxLogit = logits.Max();
        var expSum = logits.Sum(logit => MathF.Exp(logit - maxLogit));
        return logits.Select(logit => MathF.Exp(logit - maxLogit) / expSum).ToArray();
    }

    /// <summary>
    /// Normalizes the specified probability values so that their sum equals 1.
    /// </summary>
    /// <remarks>
    /// This method does not modify the input array.  
    /// The order of the values is preserved.
    /// </remarks>
    /// <param name="probabilities">
    /// An array of probability values to be normalized.  
    /// Each value should be greater than or equal to 0.  
    /// The array must not be empty.
    /// </param>
    /// <returns>
    /// A new array of floating‑point numbers containing the normalized probabilities.  
    /// The sum of all returned values is 1.
    /// </returns>
    private static float[] NormalizeProbabilities(float[] probabilities)
    {
        var sum = probabilities.Sum();
        return probabilities.Select(p => p / sum).ToArray();
    }

    /// <summary>
    /// Selects an index from the specified probability distribution using a random sample.
    /// </summary>
    /// <remarks>
    /// If the sum of the probabilities is less than 1.0 due to floating-point rounding, the last
    /// index is returned. The method assumes the input array represents a valid probability distribution.
    /// </remarks>
    /// <param name="probabilities">
    /// An array of probabilities representing the distribution to sample from. Each value should be non-negative, and
    /// the sum of all values should be 1.0.
    /// </param>
    /// <returns>The index of the selected outcome based on the provided probability distribution.</returns>
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
