using WebExpress.LLM.Inference;

namespace WebExpress.LLM.Test.Inference;

public sealed class UnitTestGreedySampling
{
    [Fact]
    public void Sample_ShouldSelectTokenWithHighestLogit()
    {
        var sampler = new GreedySampling();
        var logits = new float[] { 0.1f, 0.5f, 0.3f, 0.8f, 0.2f };

        var result = sampler.Sample(logits);

        Assert.Equal(3, result);
    }

    [Fact]
    public void Sample_ShouldBeDeterministic()
    {
        var sampler = new GreedySampling();
        var logits = new float[] { 0.1f, 0.5f, 0.3f, 0.8f, 0.2f };

        var first = sampler.Sample(logits);
        var second = sampler.Sample(logits);

        Assert.Equal(first, second);
    }

    [Fact]
    public void Sample_WithEmptyLogits_ShouldThrowArgumentException()
    {
        var sampler = new GreedySampling();
        var logits = Array.Empty<float>();

        Assert.Throws<ArgumentException>(() => sampler.Sample(logits));
    }
}
