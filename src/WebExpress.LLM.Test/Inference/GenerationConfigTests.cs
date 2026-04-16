using WebExpress.LLM.Inference;

namespace WebExpress.LLM.Test.Inference;

public sealed class GenerationConfigTests
{
    [Fact]
    public void CreateSamplingStrategy_WithoutParameters_ShouldReturnGreedySampling()
    {
        var config = new GenerationConfig();

        var strategy = config.CreateSamplingStrategy();

        Assert.IsType<GreedySampling>(strategy);
    }

    [Fact]
    public void CreateSamplingStrategy_WithTopK_ShouldReturnTopKSampling()
    {
        var config = new GenerationConfig { TopK = 5 };

        var strategy = config.CreateSamplingStrategy();

        Assert.IsType<TopKSampling>(strategy);
    }

    [Fact]
    public void CreateSamplingStrategy_WithTopP_ShouldReturnTopPSampling()
    {
        var config = new GenerationConfig { TopP = 0.9f };

        var strategy = config.CreateSamplingStrategy();

        Assert.IsType<TopPSampling>(strategy);
    }

    [Fact]
    public void CreateSamplingStrategy_WithBothTopKAndTopP_ShouldThrowInvalidOperationException()
    {
        var config = new GenerationConfig { TopK = 5, TopP = 0.9f };

        Assert.Throws<InvalidOperationException>(() => config.CreateSamplingStrategy());
    }
}
