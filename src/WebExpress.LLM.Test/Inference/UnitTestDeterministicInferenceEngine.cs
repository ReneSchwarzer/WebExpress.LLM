using WebExpress.LLM.Inference;

namespace WebExpress.LLM.Test.Inference;

/// <summary>
/// Provides unit tests for the DeterministicInferenceEngine, ensuring that token generation is reproducible.
/// </summary>
public sealed class UnitTestDeterministicInferenceEngine
{
    /// <summary>
    /// Tests that generating tokens returns a deterministic sequence.
    /// </summary>
    [Fact]
    public void GenerateTokens_ShouldReturnDeterministicSequence()
    {
        var engine = new DeterministicInferenceEngine();
        var prompt = new[] { 10, 20, 30 };

        var first = engine.GenerateTokens(prompt, 4);
        var second = engine.GenerateTokens(prompt, 4);

        Assert.Equal(first, second);
        Assert.Equal(new[] { 31, 32, 33, 34 }, first);
    }
}
