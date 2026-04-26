using System.Collections.Generic;
using WebExpress.LLM.Gemma;
using WebExpress.LLM.Model;

namespace WebExpress.LLM.Test.Gemma;

/// <summary>
/// Verifies the MoE feed-forward branch wiring matches vLLM's
/// <c>Gemma4DecoderLayer.forward</c>: the unsuffixed
/// <c>pre_feedforward_layernorm</c> belongs to the dense MLP branch, while
/// <c>pre_feedforward_layernorm_2</c> drives the MoE branch (the
/// <c>_1</c>/<c>_2</c> post norms follow the same convention).
/// </summary>
public sealed class UnitTestGemma4ModelMoeNormWiring
{
    /// <summary>
    /// Stubs the MoE expert projections to all-zero so the MoE branch
    /// contributes nothing. Then changing
    /// <c>pre_feedforward_layernorm.weight</c> (which the dense MLP uses)
    /// must change the forward output, while changing
    /// <c>pre_feedforward_layernorm_2.weight</c> (used by the now-zero MoE
    /// branch) must not.
    /// </summary>
    [Fact]
    public void Forward_PreNormWiring_DenseUsesUnsuffixedNorm()
    {
        var preferDenseLogits = RunWithNormPattern(
            preDenseConst: 1.0f,
            preMoeConst: 1.0f);

        var alteredDenseNormLogits = RunWithNormPattern(
            preDenseConst: 0.0f,   // zero out dense pre-norm → dense branch input is zero
            preMoeConst: 1.0f);

        var alteredMoeNormLogits = RunWithNormPattern(
            preDenseConst: 1.0f,
            preMoeConst: 0.0f);    // zero out MoE pre-norm → MoE input is zero

        Assert.NotEqual(preferDenseLogits, alteredDenseNormLogits);

        // MoE branch is already zeroed out by zeroed expert weights; therefore
        // changing the MoE pre-norm has no effect → outputs should match.
        Assert.Equal(preferDenseLogits, alteredMoeNormLogits);
    }

    private static float[] RunWithNormPattern(float preDenseConst, float preMoeConst)
    {
        const int hiddenSize = 4;

        // Override expert projection weights to all-zero (kills MoE contribution),
        // and override the two pre-feedforward norms to the requested constants.
        var overrides = new Dictionary<string, int[]>
        {
            // (shapes only; values come from the stub's data pattern, but for
            // these specific overrides the values are filled from the constant
            // arrays below by replacing the data after construction.)
        };

        var loader = new Gemma4StubLoader(
            numLayers: 1,
            hiddenSize: hiddenSize,
            numQueryHeads: 2,
            numKvHeads: 1,
            headDim: 2,
            numExperts: 2,
            moeIntermediate: 4,
            intermediateSize: 4,
            vocabSize: 8,
            layerTypes: ["sliding_attention"]);

        var wrapper = new ConstantTensorLoader(loader, new Dictionary<string, float[]>
        {
            // Zero out all MoE expert weights so the MoE branch contributes 0.
            ["model.language_model.layers.0.experts.gate_up_proj"] =
                new float[2 * (2 * 4) * hiddenSize], // numExperts * 2*moeInter * hiddenSize
            ["model.language_model.layers.0.experts.down_proj"] =
                new float[2 * hiddenSize * 4], // numExperts * hiddenSize * moeInter
            // Set the two pre-FFW norms to constant scales.
            ["model.language_model.layers.0.pre_feedforward_layernorm.weight"] =
                FillVector(hiddenSize, preDenseConst),
            ["model.language_model.layers.0.pre_feedforward_layernorm_2.weight"] =
                FillVector(hiddenSize, preMoeConst),
        });

        var config = new ModelConfiguration
        {
            TieWordEmbeddings = true,
            TextConfig = new TextConfig
            {
                HiddenSize = hiddenSize,
                NumberOfLayers = 1,
                NumberOfAttentionHeads = 2,
                NumberOfKeyValueHeads = 1,
                HeadDimension = 2,
                RmsNormEpsilon = 1e-6f,
                SlidingWindow = 8,
                EnableMoeBlock = true,
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

        var model = new Gemma4Model(config, wrapper);
        return model.Forward([0, 1, 2]);
    }

    private static float[] FillVector(int length, float value)
    {
        var v = new float[length];

        for (var i = 0; i < length; i++)
        {
            v[i] = value;
        }

        return v;
    }
}
