using System.Collections.Generic;

namespace WebExpress.LLM.Inference;

/// <summary>
/// Defines a strategy for sampling the next token from a probability distribution over the vocabulary.
/// </summary>
public interface ISamplingStrategy
{
    /// <summary>
    /// Samples the next token ID from the given logits.
    /// </summary>
    /// <param name="logits">The raw logits (unnormalized log probabilities) for each token in the vocabulary.</param>
    /// <returns>The selected token ID.</returns>
    int Sample(IReadOnlyList<float> logits);
}
