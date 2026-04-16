using WebExpress.LLM.Inference;
using WebExpress.LLM.Model;

namespace WebExpress.LLM.Test.Inference;

public sealed class TransformerInferenceEngineTests
{
    [Fact]
    public void GenerateTokens_ShouldProduceTokensUsingModel()
    {
        var model = CreateTestModel();
        var sampler = new GreedySampling();
        var engine = new TransformerInferenceEngine(model, sampler);

        var prompt = new[] { 1, 2, 3 };
        var result = engine.GenerateTokens(prompt, maxNewTokens: 5);

        Assert.Equal(5, result.Count);
        Assert.All(result, token => Assert.InRange(token, 0, model.Configuration.VocabularySize - 1));
    }

    [Fact]
    public void GenerateTokens_ShouldBeDeterministicWithGreedySampling()
    {
        var model = CreateTestModel();
        var sampler = new GreedySampling();
        var engine = new TransformerInferenceEngine(model, sampler);

        var prompt = new[] { 1, 2, 3 };
        var first = engine.GenerateTokens(prompt, maxNewTokens: 5);
        var second = engine.GenerateTokens(prompt, maxNewTokens: 5);

        Assert.Equal(first, second);
    }

    [Fact]
    public void GenerateTokens_WithZeroMaxTokens_ShouldReturnEmpty()
    {
        var model = CreateTestModel();
        var sampler = new GreedySampling();
        var engine = new TransformerInferenceEngine(model, sampler);

        var result = engine.GenerateTokens([1, 2, 3], maxNewTokens: 0);

        Assert.Empty(result);
    }

    private static ModelDefinition CreateTestModel()
    {
        return new ModelDefinition
        {
            Configuration = new ModelConfiguration
            {
                ModelName = "test-model",
                VocabularySize = 1000,
                ContextLength = 512,
                HiddenSize = 256,
                IntermediateSize = 512,
                NumberOfLayers = 4,
                NumberOfAttentionHeads = 8,
                NumberOfKeyValueHeads = 8,
                HeadDimension = 32
            },
            Weights = [1, 2, 3, 4]
        };
    }
}
