using System;
using System.Collections.Generic;

namespace WebExpress.LLM.Inference;

/// <summary>
/// Implements greedy decoding by always selecting the token with the highest logit value.
/// </summary>
public sealed class GreedySampling : ISamplingStrategy
{
    /// <summary>
    /// Finds the index of the highest value in the specified list of logits.
    /// </summary>
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
    public int Sample(IReadOnlyList<float> logits)
    {
        ArgumentNullException.ThrowIfNull(logits);

        if (logits.Count == 0)
        {
            throw new ArgumentException("Logits must not be empty.", nameof(logits));
        }

        var maxIndex = 0;
        var maxValue = logits[0];

        for (var i = 1; i < logits.Count; i++)
        {
            if (logits[i] > maxValue)
            {
                maxValue = logits[i];
                maxIndex = i;
            }
        }

        return maxIndex;
    }
}
