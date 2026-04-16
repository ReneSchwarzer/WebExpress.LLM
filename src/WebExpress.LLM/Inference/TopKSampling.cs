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

    /// <summary>
    /// Initializes a new instance of the TopKSampling class with the specified number of elements  
    /// and an optional random seed.
    /// </summary>
    /// <param name="k">
    /// The number of top elements to consider. Must be greater than zero.
    /// </param>
    /// <param name="seed">
    /// An optional seed value for the random number generator.  
    /// If not provided, a random seed is used.
    /// </param>
    /// <exception cref="ArgumentOutOfRangeException">
    /// Thrown when <paramref name="k"/> is less than or equal to zero.
    /// </exception>
    public TopKSampling(int k, int? seed = null)
    {
        if (k <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(k), "k must be greater than zero.");
        }

        _k = k;
        _random = seed.HasValue ? new Random(seed.Value) : new Random();
    }

    /// <summary>
    /// Samples an index from the provided logits using a top-k softmax distribution.
    /// </summary>
    /// <remarks>The method selects the top-k logits, computes their softmax probabilities, and samples an
    /// index according to this distribution. The value of k is determined by the instance configuration and is limited
    /// to the number of available logits.</remarks>
    /// <param name="logits">A read-only list of logit values representing unnormalized log probabilities. Cannot be null or empty.</param>
    /// <returns>The index of the selected logit after applying top-k filtering and sampling from the resulting probability
    /// distribution.</returns>
    /// <exception cref="ArgumentException">Thrown if logits is empty.</exception>
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

    /// <summary>
    /// Computes the softmax probability distribution for the specified array of logits.
    /// </summary>
    /// <remarks>
    /// The softmax function is commonly used in machine learning applications to convert raw values (logits)  
    /// into a probability distribution. This implementation is numerically stable even for large logit values.
    /// </remarks>
    /// <param name="logits">
    /// The array of logits for which the softmax values are to be computed. Must not be null.
    /// </param>
    /// <returns>
    /// An array of floating‑point numbers representing the softmax probabilities for each element in  
    /// <paramref name="logits"/>. The sum of all returned values equals 1.
    /// </returns>
    private static float[] Softmax(float[] logits)
    {
        var maxLogit = logits.Max();
        var expSum = logits.Sum(logit => MathF.Exp(logit - maxLogit));
        return logits.Select(logit => MathF.Exp(logit - maxLogit) / expSum).ToArray();
    }

    /// <summary>
    /// Selects an index from the specified probability distribution using a random sample.
    /// </summary>
    /// <remarks>
    /// If the sum of the probabilities is less than 1.0 due to floating-point rounding, the last
    /// index is returned. The method assumes the input array is not null and contains at least one element.
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
