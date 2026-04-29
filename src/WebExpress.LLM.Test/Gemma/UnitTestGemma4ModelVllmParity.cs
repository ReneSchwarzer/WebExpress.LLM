using WebExpress.LLM.Gemma;
using WebExpress.LLM.Model;

namespace WebExpress.LLM.Test.Gemma;

/// <summary>
/// Regression tests for Gemma-4 behavior aligned with vLLM's gemma4.py.
/// </summary>
public sealed class UnitTestGemma4ModelVllmParity
{
    [Fact]
    public void Forward_FullAttentionWithoutKEqV_UsesRegularKvHeadCount()
    {
        var loader = new Gemma4StubLoader(
            numLayers: 1,
            hiddenSize: 4,
            numQueryHeads: 2,
            numKvHeads: 1,
            headDim: 2,
            numExperts: 0,
            moeIntermediate: 0,
            intermediateSize: 4,
            vocabSize: 8,
            globalHeadDim: 2,
            numGlobalKvHeads: 2,
            attentionKeyEqualsValue: false,
            layerTypes: ["full_attention"]);

        var config = new ModelConfiguration
        {
            TieWordEmbeddings = true,
            TextConfig = new TextConfig
            {
                HiddenSize = 4,
                NumberOfLayers = 1,
                NumberOfAttentionHeads = 2,
                NumberOfKeyValueHeads = 1,
                NumberOfGlobalKeyValueHeads = 2,
                HeadDimension = 2,
                GlobalHeadDimension = 2,
                RmsNormEpsilon = 1e-6f,
                SlidingWindow = 8,
                AttentionKeyEqualsValue = false,
                IntermediateSize = 4,
                VocabularySize = 8,
                LayerTypes = ["full_attention"],
                RopeParameters = new TextRopeParameters
                {
                    FullAttention = new RopeEntry { RopeTheta = 1000000f, PartialRotaryFactor = 1f }
                }
            }
        };

        var model = new Gemma4Model(config, loader);
        var logits = model.Forward([0, 1, 2]);

        Assert.Equal(8, logits.Length);
        Assert.All(logits, v => Assert.False(float.IsNaN(v) || float.IsInfinity(v)));
        Assert.Contains("model.language_model.layers.0.self_attn.v_proj.weight", loader.Requested);
    }

    [Fact]
    public void Forward_UseSecondMlpBlock_EnablesMoePath()
    {
        var loader = new Gemma4StubLoader(
            numLayers: 1,
            hiddenSize: 4,
            numQueryHeads: 2,
            numKvHeads: 1,
            headDim: 2,
            numExperts: 2,
            moeIntermediate: 4,
            intermediateSize: 4,
            vocabSize: 8,
            layerTypes: ["sliding_attention"]);

        var config = new ModelConfiguration
        {
            TieWordEmbeddings = true,
            TextConfig = new TextConfig
            {
                HiddenSize = 4,
                NumberOfLayers = 1,
                NumberOfAttentionHeads = 2,
                NumberOfKeyValueHeads = 1,
                HeadDimension = 2,
                RmsNormEpsilon = 1e-6f,
                SlidingWindow = 8,
                EnableMoeBlock = false,
                UseSecondMlpBlock = true,
                NumberOfExperts = 2,
                TopKExperts = 1,
                MoeIntermediateSize = 4,
                IntermediateSize = 4,
                VocabularySize = 8,
                LayerTypes = ["sliding_attention"],
                RopeParameters = new TextRopeParameters
                {
                    SlidingAttention = new RopeEntry { RopeTheta = 10000f, PartialRotaryFactor = 1f }
                }
            }
        };

        var model = new Gemma4Model(config, loader);
        _ = model.Forward([0, 1, 2]);

        Assert.Contains("model.language_model.layers.0.router.proj.weight", loader.Requested);
        Assert.Contains("model.language_model.layers.0.experts.gate_up_proj", loader.Requested);
        Assert.Contains("model.language_model.layers.0.experts.down_proj", loader.Requested);
    }
}
