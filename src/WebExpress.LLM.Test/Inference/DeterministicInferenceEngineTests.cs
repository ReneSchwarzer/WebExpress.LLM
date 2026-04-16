using WebExpress.LLM.Inference;

namespace WebExpress.LLM.Test.Inference;

public sealed class DeterministicInferenceEngineTests
{
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
