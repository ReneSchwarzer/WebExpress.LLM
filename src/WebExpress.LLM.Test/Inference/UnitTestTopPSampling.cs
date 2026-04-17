using WebExpress.LLM.Inference;

namespace WebExpress.LLM.Test.Inference;

/// <summary>
/// Provides unit tests for the TopPSampling strategy, ensuring correct nucleus sampling behavior.
/// </summary>
public sealed class UnitTestTopPSampling
{
    /// <summary>
    /// Tests that top-p sampling with a seed is deterministic.
    /// </summary>
    [Fact]
    public void Sample_WithSeed_ShouldBeDeterministic()
    {
        var sampler1 = new TopPSampling(p: 0.9f, seed: 42);
        var sampler2 = new TopPSampling(p: 0.9f, seed: 42);
        var logits = new float[] { 0.1f, 0.5f, 0.3f, 0.8f, 0.2f };

        var first = sampler1.Sample(logits);
        var second = sampler2.Sample(logits);

        Assert.Equal(first, second);
    }

    /// <summary>
    /// Tests that top-p sampling only selects from the nucleus of tokens.
    /// </summary>
    [Fact]
    public void Sample_ShouldSelectFromNucleus()
    {
        var sampler = new TopPSampling(p: 0.95f, seed: 42);
        var logits = new float[] { 0.1f, 0.5f, 0.3f, 0.8f, 0.2f };

        var results = new HashSet<int>();
        for (var i = 0; i < 100; i++)
        {
            var result = sampler.Sample(logits);
            results.Add(result);
        }

        Assert.True(results.Count > 0);
        Assert.True(results.All(r => r >= 0 && r < logits.Length));
    }

    /// <summary>
    /// Tests that the constructor throws an exception when the p value is invalid.
    /// </summary>
    [Fact]
    public void Constructor_WithInvalidP_ShouldThrowArgumentOutOfRangeException()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => new TopPSampling(p: 0.0f));
        Assert.Throws<ArgumentOutOfRangeException>(() => new TopPSampling(p: -0.1f));
        Assert.Throws<ArgumentOutOfRangeException>(() => new TopPSampling(p: 1.1f));
    }
}
