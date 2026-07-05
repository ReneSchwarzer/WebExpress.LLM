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
    /// Tests that creating a sampling strategy with both top-k and top-p returns the combined pipeline
    /// rather than throwing, so the two filters can be applied together.
    /// </summary>
    [Fact]
    public void CreateSamplingStrategy_WithBothTopKAndTopP_ShouldReturnCombinedSampling()
    {
        var config = new GenerationConfig { TopK = 5, TopP = 0.9f };

        var strategy = config.CreateSamplingStrategy();

        Assert.IsType<CombinedSampling>(strategy);
    }

    /// <summary>
    /// Tests that a non-default temperature routes to the combined pipeline so the temperature is honored.
    /// </summary>
    [Fact]
    public void CreateSamplingStrategy_WithTemperature_ShouldReturnCombinedSampling()
    {
        var config = new GenerationConfig { Temperature = 0.7f };

        var strategy = config.CreateSamplingStrategy();

        Assert.IsType<CombinedSampling>(strategy);
    }

    /// <summary>
    /// Tests that a single filter combined with a non-default temperature routes to the combined pipeline.
    /// </summary>
    [Fact]
    public void CreateSamplingStrategy_WithTopKAndTemperature_ShouldReturnCombinedSampling()
    {
        var config = new GenerationConfig { TopK = 5, Temperature = 0.8f };

        var strategy = config.CreateSamplingStrategy();

        Assert.IsType<CombinedSampling>(strategy);
    }

    /// <summary>
    /// Tests that the default temperature of 1.0 keeps the specialized single-filter strategies.
    /// </summary>
    [Fact]
    public void CreateSamplingStrategy_WithDefaultTemperature_ShouldKeepSpecializedStrategies()
    {
        Assert.IsType<TopKSampling>(new GenerationConfig { TopK = 5, Temperature = 1.0f }.CreateSamplingStrategy());
        Assert.IsType<TopPSampling>(new GenerationConfig { TopP = 0.9f, Temperature = 1.0f }.CreateSamplingStrategy());
    }
}
