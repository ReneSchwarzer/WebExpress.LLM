using WebExpress.LLM.Inference;

namespace WebExpress.LLM.Test.Inference;

/// <summary>
/// Provides unit tests for the GenerationConfig class, ensuring correct creation of sampling strategies.
/// </summary>
public sealed class UnitTestGenerationConfig
{
    /// <summary>
    /// Tests that creating a sampling strategy without parameters returns greedy sampling.
    /// </summary>
    [Fact]
    public void CreateSamplingStrategy_WithoutParameters_ShouldReturnGreedySampling()
    {
        var config = new GenerationConfig();

        var strategy = config.CreateSamplingStrategy();

        Assert.IsType<GreedySampling>(strategy);
    }

    /// <summary>
    /// Tests that creating a sampling strategy with top-k returns top-k sampling.
    /// </summary>
    [Fact]
    public void CreateSamplingStrategy_WithTopK_ShouldReturnTopKSampling()
    {
        var config = new GenerationConfig { TopK = 5 };

        var strategy = config.CreateSamplingStrategy();

        Assert.IsType<TopKSampling>(strategy);
    }

    /// <summary>
    /// Tests that creating a sampling strategy with top-p returns top-p sampling.
    /// </summary>
    [Fact]
    public void CreateSamplingStrategy_WithTopP_ShouldReturnTopPSampling()
    {
        var config = new GenerationConfig { TopP = 0.9f };

        var strategy = config.CreateSamplingStrategy();

        Assert.IsType<TopPSampling>(strategy);
    }

    /// <summary>
    /// Tests that creating a sampling strategy with both top-k and top-p throws an invalid operation exception.
    /// </summary>
    [Fact]
    public void CreateSamplingStrategy_WithBothTopKAndTopP_ShouldThrowInvalidOperationException()
    {
        var config = new GenerationConfig { TopK = 5, TopP = 0.9f };

        Assert.Throws<InvalidOperationException>(() => config.CreateSamplingStrategy());
    }
}
