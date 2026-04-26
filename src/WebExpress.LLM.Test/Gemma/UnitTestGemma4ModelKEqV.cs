using System.Collections.Generic;
using WebExpress.LLM.Gemma;
using WebExpress.LLM.Model;

namespace WebExpress.LLM.Test.Gemma;

/// <summary>
/// Pins vLLM's gating rule for <c>attention_k_eq_v</c>: the K-as-V weight
/// substitution applies <em>only</em> to full-attention layers. Sliding-attention
/// layers always have their own <c>v_proj.weight</c> in the checkpoint, even
/// when the model-level flag is set (matches vLLM <c>gemma4.py:567-569</c>).
/// </summary>
public sealed class UnitTestGemma4ModelKEqV
{
    /// <summary>
    /// Mixed config with one sliding and one full layer plus
    /// <c>attention_k_eq_v=true</c>. The sliding layer must request
    /// <c>v_proj.weight</c>; the full layer must NOT.
    /// </summary>
    [Fact]
    public void Forward_KEqV_OnlyAffectsFullAttentionLayers()
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
            attentionKeyEqualsValue: true,
            layerTypes: ["sliding_attention", "full_attention"]);

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
                AttentionKeyEqualsValue = true,
                IntermediateSize = 4,
                VocabularySize = 8,
                LayerTypes = ["sliding_attention", "full_attention"],
                RopeParameters = new TextRopeParameters
                {
                    SlidingAttention = new RopeEntry { RopeTheta = 10000f, PartialRotaryFactor = 1f },
                    FullAttention = new RopeEntry { RopeTheta = 1000000f, PartialRotaryFactor = 1f }
                }
            }
        };

        var model = new Gemma4Model(config, loader);
        var logits = model.Forward([0, 1, 2]);

        Assert.Equal(8, logits.Length);
        Assert.All(logits, v => Assert.False(float.IsNaN(v) || float.IsInfinity(v)));

        // Sliding layer (index 0): real v_proj must be loaded.
        Assert.Contains("model.language_model.layers.0.self_attn.v_proj.weight", loader.Requested);

        // Full layer (index 1) with k_eq_v: must NOT load a v_proj — K weight is reused.
        Assert.DoesNotContain("model.language_model.layers.1.self_attn.v_proj.weight", loader.Requested);

        // Both layers must still load k_proj.
        Assert.Contains("model.language_model.layers.0.self_attn.k_proj.weight", loader.Requested);
        Assert.Contains("model.language_model.layers.1.self_attn.k_proj.weight", loader.Requested);
    }
}
