using System;
using System.Collections.Generic;

namespace WebExpress.LLM.Inference;

public sealed class DeterministicInferenceEngine : IInferenceEngine
{
    public IReadOnlyList<int> GenerateTokens(IReadOnlyList<int> promptTokens, int maxNewTokens)
    {
        if (promptTokens is null)
        {
            throw new ArgumentNullException(nameof(promptTokens));
        }

        if (maxNewTokens < 0)
        {
            throw new ArgumentOutOfRangeException(nameof(maxNewTokens), "Token count must be greater than or equal to zero.");
        }

        if (maxNewTokens == 0)
        {
            return Array.Empty<int>();
        }

        var seed = promptTokens.Count == 0 ? 0 : promptTokens[^1] & 0xFF;
        var output = new int[maxNewTokens];

        for (var i = 0; i < maxNewTokens; i++)
        {
            output[i] = (seed + i + 1) % 256;
        }

        return output;
    }
}
