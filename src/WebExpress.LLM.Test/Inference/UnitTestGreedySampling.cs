using WebExpress.LLM.Inference;

namespace WebExpress.LLM.Test.Inference;

/// <summary>
/// Provides unit tests for the GreedySampling strategy, ensuring the highest logit is always selected.
/// </summary>
public sealed class UnitTestGreedySampling
{
    /// <summary>
    /// Tests that the sampler selects the token with the highest logit.
    /// </summary>
    [Fact]
    public void Sample_ShouldSelectTokenWithHighestLogit()
    {
        var sampler = new GreedySampling();
        var logits = new float[] { 0.1f, 0.5f, 0.3f, 0.8f, 0.2f };

        var result = sampler.Sample(logits);

        Assert.Equal(3, result);
    }

    /// <summary>
    /// Tests that the sampling process is deterministic.
    /// </summary>
    [Fact]
    public void Sample_ShouldBeDeterministic()
    {
        var sampler = new GreedySampling();
        var logits = new float[] { 0.1f, 0.5f, 0.3f, 0.8f, 0.2f };

        var first = sampler.Sample(logits);
        var second = sampler.Sample(logits);

        Assert.Equal(first, second);
    }

    /// <summary>
    /// Tests that sampling with empty logits throws an argument exception.
    /// </summary>
    [Fact]
    public void Sample_WithEmptyLogits_ShouldThrowArgumentException()
    {
        var sampler = new GreedySampling();
        var logits = Array.Empty<float>();

        Assert.Throws<ArgumentException>(() => sampler.Sample(logits));
    }
}
