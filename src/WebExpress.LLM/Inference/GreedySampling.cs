using System;
using System.Collections.Generic;

namespace WebExpress.LLM.Inference;

/// <summary>
/// Implements greedy decoding by always selecting the token with the highest logit value.
/// </summary>
public sealed class GreedySampling : ISamplingStrategy
{
    private readonly float _repetitionPenalty;

    /// <summary>
    /// Initializes a new instance of the <see cref="GreedySampling"/> class with an optional repetition penalty.
    /// </summary>
    /// <param name="repetitionPenalty">
    /// The factor used to penalize repeated tokens during text generation. Must be greater than 0.0.
    /// A value of 1.0 disables the penalty; higher values reduce the likelihood of repetition.
    /// </param>
    /// <exception cref="ArgumentOutOfRangeException">
    /// Thrown when <paramref name="repetitionPenalty"/> is less than or equal to 0.0.
    /// </exception>
    public GreedySampling(float repetitionPenalty = 1.0f)
    {
        if (repetitionPenalty <= 0.0f)
        {
            throw new ArgumentOutOfRangeException(nameof(repetitionPenalty), "Repetition penalty must be greater than zero.");
        }

        _repetitionPenalty = repetitionPenalty;
    }

    /// <summary>
    /// Finds the index of the highest value in the specified list of logits.
    /// </summary>
    /// <remarks>
    /// Selects the index of the maximum value in the input logits (greedy argmax sampling).
    /// This method is typically used in language model decoding to deterministically choose the most likely next token.
    /// Returns the first index in case of ties.
    /// </remarks>
    /// <param name="logits">
    /// The read‑only list of floating‑point numbers from which the index of the maximum value is determined.  
    /// Must not be null or empty.
    /// </param>
    /// <returns>
    /// The index of the highest value in <paramref name="logits"/>.
    /// </returns>
    /// <exception cref="ArgumentException">
    /// Thrown when <paramref name="logits"/> is empty.
    /// </exception>
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

        var maxIndex = 0;
        var maxValue = AdjustForRepetition(logits[0], 0, seen);

        for (var i = 1; i < logits.Count; i++)
        {
            var value = AdjustForRepetition(logits[i], i, seen);

            if (value > maxValue)
            {
                maxValue = value;
                maxIndex = i;
            }
        }

        return maxIndex;
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
}
