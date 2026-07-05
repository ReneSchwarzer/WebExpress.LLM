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
    /// Tests that the repetition penalty correctly penalizes tokens seen in the context.
    /// </summary>
    [Fact]
    public void Sample_WithRepetitionPenalty_ShouldPenalizeSeenTokens()
    {
        // Use a penalty > 1.0 to ensure effect
        var sampler = new GreedySampling(repetitionPenalty: 2.0f);
        var logits = new float[] { 10.0f, 5.0f }; // Token 0 is much higher
        var context = new List<int> { 0 };       // Token 0 was already seen

        // With penalty 2.0: 
        // token 0 (positive) -> 10 / 2 = 5.0
        // token 1 (not seen) -> 5.0
        // In case of tie, GreedySampling returns first index? No, let's make it clear.
        
        var logits2 = new float[] { 6.0f, 5.0f }; 
        // token 0 -> 6 / 2 = 3.0
        // token 1 -> 5.0
        // Now token 1 should win.

        var result = sampler.Sample(logits2, context);

        Assert.Equal(1, result);
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
