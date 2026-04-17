using WebExpress.LLM.Inference;

namespace WebExpress.LLM.Test.Inference;

public sealed class UnitTestTopPSampling
{
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

    [Fact]
    public void Constructor_WithInvalidP_ShouldThrowArgumentOutOfRangeException()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => new TopPSampling(p: 0.0f));
        Assert.Throws<ArgumentOutOfRangeException>(() => new TopPSampling(p: -0.1f));
        Assert.Throws<ArgumentOutOfRangeException>(() => new TopPSampling(p: 1.1f));
    }
}
