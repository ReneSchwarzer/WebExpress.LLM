using System;
using System.Collections.Generic;

namespace WebExpress.LLM.Inference;

/// <summary>
/// Implements nucleus (top-p) sampling by selecting from the smallest set of tokens whose cumulative
/// probability exceeds the threshold p.
/// </summary>
public sealed class TopPSampling : ISamplingStrategy
{
    private readonly float _p;
    private readonly Random _random;
    private readonly float _repetitionPenalty;

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
    public TopPSampling(float p, int? seed = null, float repetitionPenalty = 1.0f)
    {
        if (p <= 0.0f || p > 1.0f)
        {
            throw new ArgumentOutOfRangeException(nameof(p), "p must be in the range (0, 1].");
        }

        if (repetitionPenalty <= 0.0f)
        {
            throw new ArgumentOutOfRangeException(nameof(repetitionPenalty), "Repetition penalty must be greater than zero.");
        }

        _p = p;
        _random = seed.HasValue ? new Random(seed.Value) : new Random();
        _repetitionPenalty = repetitionPenalty;
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
    public int Sample(IReadOnlyList<float> logits, IReadOnlyList<int> contextTokens = null)
    {
        ArgumentNullException.ThrowIfNull(logits);

        if (logits.Count == 0)
        {
            throw new ArgumentException("Logits must not be empty.", nameof(logits));
        }

        var seen = _repetitionPenalty > 1.0f && contextTokens != null && contextTokens.Count > 0
            ? new HashSet<int>(contextTokens)
            : null;

        var sortedIndices = new (float logit, int index)[logits.Count];
        for (var i = 0; i < logits.Count; i++)
        {
            sortedIndices[i] = (AdjustForRepetition(logits[i], i, seen), i);
        }

        Array.Sort(sortedIndices, static (left, right) =>
        {
            var byLogit = right.logit.CompareTo(left.logit);
            return byLogit != 0 ? byLogit : left.index.CompareTo(right.index);
        });

        var sortedLogits = new float[sortedIndices.Length];
        for (var i = 0; i < sortedIndices.Length; i++)
        {
            sortedLogits[i] = sortedIndices[i].logit;
        }

        var probabilities = Softmax(sortedLogits);

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

        var nucleusProbabilities = new float[nucleusSize];
        Array.Copy(probabilities, nucleusProbabilities, nucleusSize);

        var normalizedProbabilities = NormalizeProbabilities(nucleusProbabilities);
        var selectedIndex = SampleFromDistribution(normalizedProbabilities);

        return sortedIndices[selectedIndex].index;
    }

    private float AdjustForRepetition(float logit, int tokenId, HashSet<int> seen)
    {
        if (seen == null || !seen.Contains(tokenId) || _repetitionPenalty == 1.0f)
        {
            return logit;
        }

        return logit < 0.0f
            ? logit * _repetitionPenalty
            : logit / _repetitionPenalty;
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
        var maxLogit = float.NegativeInfinity;
        for (var i = 0; i < logits.Length; i++)
        {
            if (logits[i] > maxLogit)
            {
                maxLogit = logits[i];
            }
        }

        var expSum = 0.0f;
        var probabilities = new float[logits.Length];
        for (var i = 0; i < logits.Length; i++)
        {
            var exp = MathF.Exp(logits[i] - maxLogit);
            probabilities[i] = exp;
            expSum += exp;
        }

        for (var i = 0; i < probabilities.Length; i++)
        {
            probabilities[i] /= expSum;
        }

        return probabilities;
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
        var sum = 0.0f;
        for (var i = 0; i < probabilities.Length; i++)
        {
            sum += probabilities[i];
        }

        var normalized = new float[probabilities.Length];
        for (var i = 0; i < probabilities.Length; i++)
        {
            normalized[i] = probabilities[i] / sum;
        }

        return normalized;
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
