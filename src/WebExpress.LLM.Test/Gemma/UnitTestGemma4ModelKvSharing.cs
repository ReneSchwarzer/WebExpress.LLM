using WebExpress.LLM.Gemma;
using WebExpress.LLM.Model;

namespace WebExpress.LLM.Test.Gemma;

/// <summary>
/// Pins the cross-layer KV-cache sharing pipeline (vLLM
/// <c>num_kv_shared_layers &gt; 0</c>): the trailing shared layers must reuse
/// K/V from the preceding layer of the same attention type and therefore must
/// not request their own <c>k_proj</c>/<c>v_proj</c>/<c>k_norm</c> tensors.
/// </summary>
public sealed class UnitTestGemma4ModelKvSharing
{
    /// <summary>
    /// Two sliding layers; the second is KV-shared. Layer 0 must load
    /// <c>k_proj</c>/<c>v_proj</c>/<c>k_norm</c>; layer 1 must NOT. The
    /// forward pass must still produce valid logits.
    /// </summary>
    [Fact]
    public void Forward_KvSharedLayer_SkipsKvProjections()
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
                NumberOfKvSharedLayers = 1,
                IntermediateSize = 4,
                VocabularySize = 8,
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

        // Layer 0 (the source layer): owns its K/V projections.
        Assert.Contains("model.language_model.layers.0.self_attn.k_proj.weight", loader.Requested);
        Assert.Contains("model.language_model.layers.0.self_attn.v_proj.weight", loader.Requested);

        // Layer 1 (KV-shared): must reuse layer 0's cached K/V — no own K/V projections.
        Assert.DoesNotContain("model.language_model.layers.1.self_attn.k_proj.weight", loader.Requested);
        Assert.DoesNotContain("model.language_model.layers.1.self_attn.v_proj.weight", loader.Requested);
        Assert.DoesNotContain("model.language_model.layers.1.self_attn.k_norm.weight", loader.Requested);

        // Both layers still have their own Q projection.
        Assert.Contains("model.language_model.layers.0.self_attn.q_proj.weight", loader.Requested);
        Assert.Contains("model.language_model.layers.1.self_attn.q_proj.weight", loader.Requested);
    }
}
