using System.Collections.Generic;
using System.Linq;
using WebExpress.LLM.Gemma;
using WebExpress.LLM.Model;
using WebExpress.LLM.SafeTensors;

namespace WebExpress.LLM.Test.Gemma;

/// <summary>
/// Integration test that exercises the complete Gemma4Model forward pass through
/// the MoE + dense-shared (mlp2) feed-forward dispatch. A stubbed
/// <see cref="ISafeTensorLoader"/> returns deterministic small tensors for every
/// weight name the model requests, which additionally pins the weight-name
/// contract: if the model starts asking for a renamed or new tensor it will
/// fall through to the default zero tensor and a dedicated assertion catches it.
/// </summary>
public sealed class UnitTestGemma4ModelMoeDispatch
{
    /// <summary>
    /// Runs a full forward pass with MoE enabled and two transformer layers, and
    /// verifies both the output shape and the set of tensor names that were
    /// requested from the loader.
    /// </summary>
    [Fact]
    public void Forward_WithMoeEnabled_ShouldProduceCorrectShapeAndRequestExpectedWeights()
    {
        var numLayers = 2;
        var hiddenSize = 4;
        var numQueryHeads = 2;
        var numKvHeads = 1;
        var headDim = 2;
        var numExperts = 2;
        var topKExperts = 1;
        var moeIntermediate = 4;
        var intermediateSize = 4;
        var vocabSize = 8;

        var loader = new RecordingStubLoader(
            numLayers, hiddenSize, numQueryHeads, numKvHeads, headDim,
            numExperts, moeIntermediate, intermediateSize, vocabSize);

        var config = new ModelConfiguration
        {
            TieWordEmbeddings = true,
            TextConfig = new TextConfig
            {
                HiddenSize = hiddenSize,
                NumberOfLayers = numLayers,
                NumberOfAttentionHeads = numQueryHeads,
                NumberOfKeyValueHeads = numKvHeads,
                HeadDimension = headDim,
                RmsNormEpsilon = 1e-6f,
                SlidingWindow = 8,
                AttentionKeyEqualsValue = false,
                EnableMoeBlock = true,
                NumberOfExperts = numExperts,
                TopKExperts = topKExperts,
                MoeIntermediateSize = moeIntermediate,
                IntermediateSize = intermediateSize,
                VocabularySize = vocabSize,
                LayerTypes = Enumerable.Repeat("sliding_attention", numLayers).ToList(),
                RopeParameters = new TextRopeParameters
                {
                    SlidingAttention = new RopeEntry { RopeTheta = 10000f, PartialRotaryFactor = 1f }
                }
            }
        };

        var model = new Gemma4Model(config, loader);
        var logits = model.Forward([0, 1, 2]);

        Assert.Equal(vocabSize, logits.Length);

        foreach (var v in logits)
        {
            Assert.False(float.IsNaN(v));
            Assert.False(float.IsInfinity(v));
        }

        // Weight-name contract: every per-layer tensor the implementation relies
        // on must have been requested. If one of these names silently changes we
        // want the test to break.
        for (var layer = 0; layer < numLayers; layer++)
        {
            var prefix = $"model.language_model.layers.{layer}";
            Assert.Contains($"{prefix}.input_layernorm.weight", loader.Requested);
            Assert.Contains($"{prefix}.post_attention_layernorm.weight", loader.Requested);
            Assert.Contains($"{prefix}.self_attn.q_proj.weight", loader.Requested);
            Assert.Contains($"{prefix}.self_attn.k_proj.weight", loader.Requested);
            Assert.Contains($"{prefix}.self_attn.v_proj.weight", loader.Requested);
            Assert.Contains($"{prefix}.self_attn.o_proj.weight", loader.Requested);
            Assert.Contains($"{prefix}.pre_feedforward_layernorm.weight", loader.Requested);
            Assert.Contains($"{prefix}.pre_feedforward_layernorm_2.weight", loader.Requested);
            Assert.Contains($"{prefix}.post_feedforward_layernorm_1.weight", loader.Requested);
            Assert.Contains($"{prefix}.post_feedforward_layernorm_2.weight", loader.Requested);
            Assert.Contains($"{prefix}.post_feedforward_layernorm.weight", loader.Requested);
            Assert.Contains($"{prefix}.router.proj.weight", loader.Requested);
            Assert.Contains($"{prefix}.experts.gate_up_proj", loader.Requested);
            Assert.Contains($"{prefix}.experts.down_proj", loader.Requested);
            Assert.Contains($"{prefix}.mlp.gate_proj.weight", loader.Requested);
            Assert.Contains($"{prefix}.mlp.up_proj.weight", loader.Requested);
            Assert.Contains($"{prefix}.mlp.down_proj.weight", loader.Requested);
        }

        Assert.Contains("model.language_model.embed_tokens.weight", loader.Requested);
        Assert.Contains("model.language_model.norm.weight", loader.Requested);
    }

    /// <summary>
    /// Deterministic in-memory <see cref="ISafeTensorLoader"/> that returns small
    /// synthetic tensors whose shape is inferred from the requested tensor name.
    /// It also records every name that was requested so the test can verify the
    /// weight-name contract without having to fake file I/O.
    /// </summary>
    private sealed class RecordingStubLoader : ISafeTensorLoader
    {
        private readonly int _numLayers;
        private readonly int _hiddenSize;
        private readonly int _numQueryHeads;
        private readonly int _numKvHeads;
        private readonly int _headDim;
        private readonly int _numExperts;
        private readonly int _moeInter;
        private readonly int _intermediateSize;
        private readonly int _vocabSize;

        public RecordingStubLoader(
            int numLayers, int hiddenSize, int numQueryHeads, int numKvHeads, int headDim,
            int numExperts, int moeInter, int intermediateSize, int vocabSize)
        {
            _numLayers = numLayers;
            _hiddenSize = hiddenSize;
            _numQueryHeads = numQueryHeads;
            _numKvHeads = numKvHeads;
            _headDim = headDim;
            _numExperts = numExperts;
            _moeInter = moeInter;
            _intermediateSize = intermediateSize;
            _vocabSize = vocabSize;
        }

        public HashSet<string> Requested { get; } = [];

        public IReadOnlyCollection<string> TensorNames => [];

        public TensorMetadata GetMetadata(string name)
        {
            throw new KeyNotFoundException(name);
        }

        public bool ContainsTensor(string name)
        {
            return TryShape(name) is not null;
        }

        public WebExpress.LLM.Tensor.Tensor LoadTensor(string name)
        {
            Requested.Add(name);

            var shape = TryShape(name) ?? throw new KeyNotFoundException(name);
            var size = 1;

            foreach (var d in shape)
            {
                size *= d;
            }

            var data = new float[size];

            for (var i = 0; i < data.Length; i++)
            {
                // Small deterministic pattern, keeps activations in a sane range.
                data[i] = ((i * 7 + 3) % 11 - 5) * 0.05f;
            }

            return new WebExpress.LLM.Tensor.Tensor(shape, data);
        }

        /// <summary>
        /// Returns the shape for a requested tensor name, or null when the
        /// implementation does not recognise the name. The name space is
        /// limited to weights used by <see cref="Gemma4Model"/> when MoE is
        /// enabled — missing names mean the loader does not provide that
        /// tensor and <see cref="ISafeTensorLoader.TryLoadTensor(string)"/>
        /// should yield null.
        /// </summary>
        private int[]? TryShape(string name)
        {
            if (name == "model.language_model.embed_tokens.weight")
            {
                return [_vocabSize, _hiddenSize];
            }

            if (name == "model.language_model.norm.weight")
            {
                return [_hiddenSize];
            }

            if (!name.StartsWith("model.language_model.layers."))
            {
                return null;
            }

            var remainder = name["model.language_model.layers.".Length..];
            var dotIndex = remainder.IndexOf('.');

            if (dotIndex <= 0)
            {
                return null;
            }

            var layerStr = remainder[..dotIndex];

            if (!int.TryParse(layerStr, out var layerIndex) ||
                layerIndex < 0 || layerIndex >= _numLayers)
            {
                return null;
            }

            var suffix = remainder[(dotIndex + 1)..];

            return suffix switch
            {
                "input_layernorm.weight" => [_hiddenSize],
                "post_attention_layernorm.weight" => [_hiddenSize],
                "pre_feedforward_layernorm.weight" => [_hiddenSize],
                "pre_feedforward_layernorm_2.weight" => [_hiddenSize],
                "post_feedforward_layernorm.weight" => [_hiddenSize],
                "post_feedforward_layernorm_1.weight" => [_hiddenSize],
                "post_feedforward_layernorm_2.weight" => [_hiddenSize],
                "self_attn.q_norm.weight" => [_headDim],
                "self_attn.k_norm.weight" => [_headDim],
                "self_attn.q_proj.weight" => [_numQueryHeads * _headDim, _hiddenSize],
                "self_attn.k_proj.weight" => [_numKvHeads * _headDim, _hiddenSize],
                "self_attn.v_proj.weight" => [_numKvHeads * _headDim, _hiddenSize],
                "self_attn.o_proj.weight" => [_hiddenSize, _numQueryHeads * _headDim],
                "mlp.gate_proj.weight" => [_intermediateSize, _hiddenSize],
                "mlp.up_proj.weight" => [_intermediateSize, _hiddenSize],
                "mlp.down_proj.weight" => [_hiddenSize, _intermediateSize],
                "router.proj.weight" => [_numExperts, _hiddenSize],
                "experts.gate_up_proj" => [_numExperts, 2 * _moeInter, _hiddenSize],
                "experts.down_proj" => [_numExperts, _hiddenSize, _moeInter],
                "layer_scalar" => [1],
                _ => null
            };
        }
    }
}
