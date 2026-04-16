using WebExpress.LLM.Inference;

namespace WebExpress.LLM.Test.Inference;

public sealed class TopKSamplingTests
{
    [Fact]
    public void Sample_WithSeed_ShouldBeDeterministic()
    {
        var sampler1 = new TopKSampling(k: 3, seed: 42);
        var sampler2 = new TopKSampling(k: 3, seed: 42);
        var logits = new float[] { 0.1f, 0.5f, 0.3f, 0.8f, 0.2f };

        var first = sampler1.Sample(logits);
        var second = sampler2.Sample(logits);

        Assert.Equal(first, second);
    }

    [Fact]
    public void Sample_ShouldSelectFromTopKTokens()
    {
        var sampler = new TopKSampling(k: 3, seed: 42);
        var logits = new float[] { 0.1f, 0.5f, 0.3f, 0.8f, 0.2f };

        var results = new HashSet<int>();
        for (var i = 0; i < 100; i++)
        {
            var result = sampler.Sample(logits);
            results.Add(result);
        }

        var topKIndices = new[] { 1, 2, 3 };
        Assert.True(results.All(r => topKIndices.Contains(r)));
    }

    [Fact]
    public void Constructor_WithInvalidK_ShouldThrowArgumentOutOfRangeException()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => new TopKSampling(k: 0));
        Assert.Throws<ArgumentOutOfRangeException>(() => new TopKSampling(k: -1));
    }
}
