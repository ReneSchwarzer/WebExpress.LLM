using System;
using System.Collections.Generic;

namespace WebExpress.LLM.Inference;

/// <summary>
/// Implements a configurable sampling pipeline that applies, in order, a repetition penalty,
/// temperature scaling, an optional top-k filter, and an optional top-p (nucleus) filter before
/// sampling the next token. This mirrors the standard decoding pipeline used by common inference
/// stacks and, unlike <see cref="TopKSampling"/> and <see cref="TopPSampling"/>, allows top-k and
/// top-p to be combined and honors the temperature setting.
/// </summary>
/// <remarks>
/// The filters are complementary: top-k caps the absolute number of candidate tokens, while top-p
/// caps them by cumulative probability. When both are set, top-k is applied first and top-p operates
/// on the renormalized distribution of the surviving tokens. A temperature of <c>0</c> (or lower)
/// degenerates to greedy (argmax) decoding.
/// </remarks>
public sealed class CombinedSampling : ISamplingStrategy
{
    private readonly float _temperature;
    private readonly int? _topK;
    private readonly float? _topP;
    private readonly Random _random;
    private readonly float _repetitionPenalty;

    /// <summary>
    /// Initializes a new instance of the <see cref="CombinedSampling"/> class.
    /// </summary>
    /// <param name="temperature">
    /// The temperature applied to the logits before sampling. Must be greater than or equal to 0.
    /// A value of 1.0 leaves the distribution unchanged; values below 1.0 sharpen it and values above
    /// 1.0 flatten it. A value of 0 selects the highest-probability token deterministically (greedy).
    /// </param>
    /// <param name="topK">
    /// The optional number of highest-probability tokens to keep. When provided, must be greater than
    /// zero. When null, no top-k filtering is applied.
    /// </param>
    /// <param name="topP">
    /// The optional cumulative-probability threshold for nucleus sampling. When provided, must be in
    /// the range (0, 1]. When null, no top-p filtering is applied.
    /// </param>
    /// <param name="seed">
    /// An optional seed value for the random number generator. If not provided, a random seed is used.
    /// </param>
    /// <param name="repetitionPenalty">
    /// The factor used to penalize already-seen tokens. Must be greater than 0.0; a value of 1.0
    /// disables the penalty.
    /// </param>
    /// <exception cref="ArgumentOutOfRangeException">
    /// Thrown when <paramref name="temperature"/> is negative, <paramref name="topK"/> is not greater
    /// than zero, <paramref name="topP"/> is outside (0, 1], or <paramref name="repetitionPenalty"/>
    /// is not greater than zero.
    /// </exception>
    public CombinedSampling(float temperature, int? topK, float? topP, int? seed = null, float repetitionPenalty = 1.0f)
    {
        if (temperature < 0.0f)
        {
            throw new ArgumentOutOfRangeException(nameof(temperature), "Temperature must be greater than or equal to zero.");
        }

        if (topK.HasValue && topK.Value <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(topK), "k must be greater than zero.");
        }

        if (topP.HasValue && (topP.Value <= 0.0f || topP.Value > 1.0f))
        {
            throw new ArgumentOutOfRangeException(nameof(topP), "p must be in the range (0, 1].");
        }

        if (repetitionPenalty <= 0.0f)
        {
            throw new ArgumentOutOfRangeException(nameof(repetitionPenalty), "Repetition penalty must be greater than zero.");
        }

        _temperature = temperature;
        _topK = topK;
        _topP = topP;
        _random = seed.HasValue ? new Random(seed.Value) : new Random();
        _repetitionPenalty = repetitionPenalty;
    }

    /// <summary>
    /// Samples an index from the provided logits by applying the repetition penalty, temperature
    /// scaling, top-k, and top-p filters in sequence.
    /// </summary>
    /// <param name="logits">A read-only list of logit values representing unnormalized log probabilities. Cannot be null or empty.</param>
    /// <param name="contextTokens">Optional tokens seen so far, used for repetition-aware sampling.</param>
    /// <returns>The index of the selected token.</returns>
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

        // Apply the repetition penalty while keeping each logit's original index.
        var scored = new (float logit, int index)[logits.Count];
        for (var i = 0; i < logits.Count; i++)
        {
            scored[i] = (AdjustForRepetition(logits[i], i, seen), i);
        }

        // A temperature of zero (or lower) degenerates to greedy decoding.
        if (_temperature <= 0.0f)
        {
            return ArgMax(scored);
        }

        // Temperature scaling (1.0 is a no-op).
        if (_temperature != 1.0f)
        {
            for (var i = 0; i < scored.Length; i++)
            {
                scored[i].logit /= _temperature;
            }
        }

        Array.Sort(scored, static (left, right) =>
        {
            var byLogit = right.logit.CompareTo(left.logit);
            return byLogit != 0 ? byLogit : left.index.CompareTo(right.index);
        });

        // Top-k: keep at most k of the highest-scoring tokens.
        var count = _topK.HasValue ? Math.Min(_topK.Value, scored.Length) : scored.Length;

        var topLogits = new float[count];
        for (var i = 0; i < count; i++)
        {
            topLogits[i] = scored[i].logit;
        }

        var probabilities = Softmax(topLogits);

        // Top-p: keep the smallest prefix whose cumulative probability reaches the threshold.
        var keep = count;
        if (_topP.HasValue)
        {
            var cumulative = 0.0f;
            keep = 0;
            for (var i = 0; i < probabilities.Length; i++)
            {
                cumulative += probabilities[i];
                keep++;

                if (cumulative >= _topP.Value)
                {
                    break;
                }
            }
        }

        var nucleus = new float[keep];
        Array.Copy(probabilities, nucleus, keep);

        var normalized = NormalizeProbabilities(nucleus);
        var selected = SampleFromDistribution(normalized);

        return scored[selected].index;
    }

    private static int ArgMax((float logit, int index)[] scored)
    {
        var bestIndex = scored[0].index;
        var bestLogit = scored[0].logit;

        for (var i = 1; i < scored.Length; i++)
        {
            if (scored[i].logit > bestLogit)
            {
                bestLogit = scored[i].logit;
                bestIndex = scored[i].index;
            }
        }

        return bestIndex;
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
    /// Computes a numerically stable softmax probability distribution for the specified logits.
    /// </summary>
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
