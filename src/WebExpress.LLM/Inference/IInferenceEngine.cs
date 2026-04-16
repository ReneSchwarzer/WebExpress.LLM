using System.Collections.Generic;

namespace WebExpress.LLM.Inference;

/// <summary>
/// Defines an interface for generating new token sequences based on a given prompt using an inference engine.
/// </summary>
/// <remarks>Implementations of this interface typically perform machine learning inference to extend or complete
/// tokenized input sequences. The behavior and quality of the generated tokens depend on the underlying model and
/// configuration.</remarks>
public interface IInferenceEngine
{
    /// <summary>
    /// Generates a sequence of new tokens based on the provided input tokens and the maximum number  
    /// of new tokens to produce.
    /// </summary>
    /// <param name="promptTokens">
    /// The list of input tokens used as the starting point for generation. Must not be null.
    /// </param>
    /// <param name="maxNewTokens">
    /// The maximum number of new tokens to generate. Must be greater than 0.
    /// </param>
    /// <returns>
    /// A read‑only list of integers containing the generated tokens.  
    /// The list may be empty if no new tokens are produced.
    /// </returns>
    IReadOnlyList<int> GenerateTokens(IReadOnlyList<int> promptTokens, int maxNewTokens);
}
