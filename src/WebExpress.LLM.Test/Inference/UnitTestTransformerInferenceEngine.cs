using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using WebExpress.LLM.Inference;
using WebExpress.LLM.Model;

namespace WebExpress.LLM.Test.Inference;

/// <summary>
/// Provides unit tests for the TransformerInferenceEngine class.
/// </summary>
public sealed class UnitTestTransformerInferenceEngine
{
    [Fact]
    public void GenerateTokens_FallbackMode_ShouldWorkAndBeDeterministic()
    {
        var config = new ModelConfiguration
        {
            VocabularySize = 256,
            ContextLength = 128,
            HiddenSize = 64,
            NumberOfLayers = 2,
            NumberOfAttentionHeads = 2,
            NumberOfKeyValueHeads = 1,
            HeadDimension = 8
        };

        var modelDef = new ModelDefinition
        {
            Configuration = config
        };

        var samplingStrategy = new GreedySampling();
        var engine = new TransformerInferenceEngine(modelDef, samplingStrategy);

        var prompt = new[] { 10, 20, 30 };
        var firstResult = engine.GenerateTokens(prompt, 5);
        var secondResult = engine.GenerateTokens(prompt, 5);

        Assert.Equal(5, firstResult.Count);
        Assert.Equal(firstResult, secondResult);
    }

    [Fact]
    public async Task GenerateTokensAsync_FallbackMode_ShouldWorkAndBeDeterministic()
    {
        var config = new ModelConfiguration
        {
            VocabularySize = 256,
            ContextLength = 128,
            HiddenSize = 64,
            NumberOfLayers = 2,
            NumberOfAttentionHeads = 2,
            NumberOfKeyValueHeads = 1,
            HeadDimension = 8
        };

        var modelDef = new ModelDefinition
        {
            Configuration = config
        };

        var samplingStrategy = new GreedySampling();
        var engine = new TransformerInferenceEngine(modelDef, samplingStrategy);

        var prompt = new[] { 10, 20, 30 };
        var firstResult = new List<int>();
        await foreach (var token in engine.GenerateTokensAsync(prompt, 5))
        {
            firstResult.Add(token);
        }

        var secondResult = new List<int>();
        await foreach (var token in engine.GenerateTokensAsync(prompt, 5))
        {
            secondResult.Add(token);
        }

        Assert.Equal(5, firstResult.Count);
        Assert.Equal(firstResult, secondResult);
    }
}
