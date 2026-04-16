using System;
using System.Collections.Generic;

namespace WebExpress.LLM.Inference;

/// <summary>
/// Provides a deterministic inference engine that generates new token sequences based on a given prompt.
/// </summary>
/// <remarks>
/// This engine always produces the same output sequence for an identical prompt and token count.  
/// It is suitable for scenarios where predictability and reproducibility of token generation are required.
/// </remarks>
public sealed class DeterministicInferenceEngine : IInferenceEngine
{
    /// <summary>
    /// Generates a sequence of new token values based on the provided prompt tokens and the specified maximum number of
    /// tokens.
    /// </summary>
    /// <param name="promptTokens">The list of prompt tokens used as the basis for generating new tokens. Cannot be null.</param>
    /// <param name="maxNewTokens">The maximum number of new tokens to generate. Must be greater than or equal to zero.</param>
    /// <returns>A read-only list containing the generated token values. Returns an empty list if maxNewTokens is zero.</returns>
    /// <exception cref="ArgumentNullException">Thrown if promptTokens is null.</exception>
    /// <exception cref="ArgumentOutOfRangeException">Thrown if maxNewTokens is less than zero.</exception>
    public IReadOnlyList<int> GenerateTokens(IReadOnlyList<int> promptTokens, int maxNewTokens)
    {
        ArgumentNullException.ThrowIfNull(promptTokens);

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
