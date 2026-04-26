using WebExpress.LLM.Gemma;
using WebExpress.LLM.Model;

namespace WebExpress.LLM.Test.Gemma;

/// <summary>
/// Verifies the Per-Layer Embedding (PLE) pipeline runs end-to-end and
/// requests every PLE-related tensor that vLLM's
/// <c>Gemma4SelfDecoderLayers</c> + <c>Gemma4DecoderLayer</c> consume when
/// <c>hidden_size_per_layer_input &gt; 0</c>.
/// </summary>
public sealed class UnitTestGemma4ModelPle
{
    /// <summary>
    /// Builds a 2-layer non-MoE model with PLE enabled and asserts both the
    /// model-level PLE tensors and the per-layer PLE tensors are requested.
    /// </summary>
    [Fact]
    public void Forward_WithPle_LoadsAllPleTensors()
    {
        var loader = new Gemma4StubLoader(
            numLayers: 2,
            hiddenSize: 4,
            numQueryHeads: 2,
            numKvHeads: 1,
            headDim: 2,
            numExperts: 0,
            moeIntermediate: 0,
            intermediateSize: 4,
            vocabSize: 8,
            hiddenSizePerLayerInput: 2,
            vocabSizePerLayerInput: 8,
            layerTypes: ["sliding_attention", "sliding_attention"]);

        var config = new ModelConfiguration
        {
            TieWordEmbeddings = true,
            TextConfig = new TextConfig
            {
                HiddenSize = 4,
                NumberOfLayers = 2,
                NumberOfAttentionHeads = 2,
                NumberOfKeyValueHeads = 1,
                HeadDimension = 2,
                RmsNormEpsilon = 1e-6f,
                SlidingWindow = 8,
                IntermediateSize = 4,
                VocabularySize = 8,
                HiddenSizePerLayerInput = 2,
                VocabSizePerLayerInput = 8,
                LayerTypes = ["sliding_attention", "sliding_attention"],
                RopeParameters = new TextRopeParameters
                {
                    SlidingAttention = new RopeEntry { RopeTheta = 10000f, PartialRotaryFactor = 1f }
                }
            }
        };

        var model = new Gemma4Model(config, loader);
        var logits = model.Forward([0, 1, 2]);

        Assert.Equal(8, logits.Length);
        Assert.All(logits, v => Assert.False(float.IsNaN(v) || float.IsInfinity(v)));

        Assert.Contains("model.language_model.embed_tokens_per_layer.weight", loader.Requested);
        Assert.Contains("model.language_model.per_layer_model_projection.weight", loader.Requested);
        Assert.Contains("model.language_model.per_layer_projection_norm.weight", loader.Requested);

        for (var layer = 0; layer < 2; layer++)
        {
            var prefix = $"model.language_model.layers.{layer}";
            Assert.Contains($"{prefix}.per_layer_input_gate.weight", loader.Requested);
            Assert.Contains($"{prefix}.per_layer_projection.weight", loader.Requested);
            Assert.Contains($"{prefix}.post_per_layer_input_norm.weight", loader.Requested);
        }
    }
}
