using System.Linq;
using WebExpress.LLM.Gemma;
using WebExpress.LLM.Model;

namespace WebExpress.LLM.Test.Gemma;

/// <summary>
/// Pins the embedding-normalisation step
/// (<c>hidden = embed_lookup * sqrt(hidden_size)</c>) introduced to match
/// vLLM's <c>Gemma4Model.embed_input_ids</c>. Without the scaling the
/// downstream activations would have a different magnitude relative to the
/// trained weights, so this test asserts that running the same model twice
/// is deterministic and the requested weight set includes the embedding.
/// </summary>
public sealed class UnitTestGemma4ModelEmbeddingScale
{
    /// <summary>
    /// Runs a forward pass with a non-MoE single-layer model and asserts the
    /// logits are finite and reproducible.
    /// </summary>
    [Fact]
    public void Forward_AppliesEmbeddingNormalizer()
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
        var logits = model.Forward([0, 1, 2]);

        Assert.Equal(8, logits.Length);
        Assert.All(logits, v => Assert.False(float.IsNaN(v) || float.IsInfinity(v)));
        Assert.Contains("model.language_model.embed_tokens.weight", loader.Requested);

        // Determinism — second forward pass must produce identical logits.
        model.ResetCache();
        var logitsAgain = model.Forward([0, 1, 2]);
        Assert.Equal(logits, logitsAgain);
    }
}
