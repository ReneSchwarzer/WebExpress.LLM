using System;
using System.Collections.Generic;
using System.Linq;

namespace WebExpress.LLM.Inference;

/// <summary>
/// Implements greedy decoding by always selecting the token with the highest logit value.
/// </summary>
public sealed class GreedySampling : ISamplingStrategy
{
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
