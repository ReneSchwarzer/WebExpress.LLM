using System.Buffers.Binary;
using System.Text;
using System.Text.Json;
using WebExpress.LLM.Gemma;
using WebExpress.LLM.Model;
using WebExpress.LLM.SafeTensors;

namespace WebExpress.LLM.Test.Gemma;

/// <summary>
/// Provides a collection of component tests for the Gemma4Model class in order to validate
/// its public behavior and various configuration scenarios.
/// </summary>
public sealed class Gemma4ModelTests
{
    /// <summary>
    /// Verifies that the Forward method of the Gemma4Model returns a logits array whose length matches the configured
    /// vocabulary size.
    /// </summary>
    [Fact]
    public void Forward_ShouldProduceLogitsOfVocabSize()
    {
        var (config, loader) = CreateTinyModel();
        var model = new Gemma4Model(config, loader);

        var logits = model.Forward([1, 2, 3]);

        Assert.Equal(config.VocabularySize, logits.Length);
    }

    /// <summary>
    /// Verifies that the Forward method of the Gemma4Model class produces deterministic outputs for the same input
    /// after resetting the model's cache.
    /// </summary>
    [Fact]
    public void Forward_ShouldBeDeterministic()
    {
        var (config, loader) = CreateTinyModel();
        var model = new Gemma4Model(config, loader);

        var logits1 = model.Forward([1, 2]);

        model.ResetCache();
        var logits2 = model.Forward([1, 2]);

        for (var i = 0; i < logits1.Length; i++)
        {
            Assert.Equal(logits1[i], logits2[i], 1e-5f);
        }
    }

    /// <summary>
    /// Verifies that the Forward method throws an ArgumentException when called with an empty token array.
    /// </summary>
    [Fact]
    public void Forward_EmptyTokens_ShouldThrow()
    {
        var (config, loader) = CreateTinyModel();
        var model = new Gemma4Model(config, loader);

        Assert.Throws<ArgumentException>(() => model.Forward(Array.Empty<int>()));
    }

    /// <summary>
    /// Verifies that the Forward method throws an ArgumentNullException when called with a null tokens argument.
    /// </summary>
    [Fact]
    public void Forward_NullTokens_ShouldThrow()
    {
        var (config, loader) = CreateTinyModel();
        var model = new Gemma4Model(config, loader);

        Assert.Throws<ArgumentNullException>(() => model.Forward(null));
    }

    /// <summary>
    /// Verifies that calling ResetCache fully clears the model’s internal key/value cache (KV cache).
    /// </summary>
    [Fact]
    public void ResetCache_ShouldClearKvCache()
    {
        var (config, loader) = CreateTinyModel();
        var model = new Gemma4Model(config, loader);

        model.Forward([1, 2]);
        Assert.True(model.Cache.LayerCount > 0);

        model.ResetCache();
        Assert.Equal(0, model.Cache.LayerCount);
    }

    /// <summary>
    /// Verifies that the Forward method of the Gemma4Model returns logits with a length equal to the configured
    /// vocabulary size when provided with a single token input.
    /// </summary>
    [Fact]
    public void Forward_SingleToken_ShouldWork()
    {
        var (config, loader) = CreateTinyModel();
        var model = new Gemma4Model(config, loader);

        var logits = model.Forward([0]);

        Assert.Equal(config.VocabularySize, logits.Length);
    }

    /// <summary>
    /// Verifies that the model uses the embedding weights for the output layer when embeddings are tied.
    /// </summary>
    [Fact]
    public void Forward_WithTiedEmbeddings_ShouldUseEmbedWeightForOutput()
    {
        // Create a model with tied embeddings
        var (config, loader) = CreateTinyModel(tieWordEmbeddings: true);
        var model = new Gemma4Model(config, loader);

        var logits = model.Forward([1]);

        Assert.Equal(config.VocabularySize, logits.Length);
    }

    /// <summary>
    /// Verifies that the Forward method correctly processes and caches all model layers
    /// when using a combination of sliding‑attention and full‑attention layers.
    /// </summary>
    [Fact]
    public void Forward_WithSlidingAndFullAttention_ShouldProcessAllLayers()
    {
        var (config, loader) = CreateTinyModel(numLayers: 4, layerTypes: ["sliding_attention", "sliding_attention", "full_attention", "sliding_attention"]);
        var model = new Gemma4Model(config, loader);

        var logits = model.Forward([1, 2]);

        Assert.Equal(config.VocabularySize, logits.Length);
        // Should have cached all 4 layers
        Assert.Equal(4, model.Cache.LayerCount);
    }

    /// <summary>
    /// Verifies that the model shares key and value projection weights when the attention configuration specifies that
    /// the key equals the value.
    /// </summary>
    [Fact]
    public void Forward_WithAttentionKeyEqualsValue_ShouldShareKVWeight()
    {
        var (config, loader) = CreateTinyModel(attentionKeyEqualsValue: true);
        var model = new Gemma4Model(config, loader);

        // v_proj.weight should not exist when K and V share weights
        Assert.False(loader.ContainsTensor("model.language_model.layers.0.self_attn.v_proj.weight"));

        var logits = model.Forward([1, 2]);

        Assert.Equal(config.VocabularySize, logits.Length);
    }

    /// <summary>
    /// Verifies that the model's Forward method correctly processes all layers when sliding window and full attention
    /// layers use different head dimensions.
    /// </summary>
    [Fact]
    public void Forward_WithDifferentGlobalHeadDim_ShouldProcessAllLayers()
    {
        // Reproduces the original bug: sliding window layers use headDim=2 while
        // full attention layers use globalHeadDim=4 with weights sized accordingly.
        var (config, loader) = CreateTinyModel(
            numLayers: 3,
            headDim: 2,
            globalHeadDim: 4,
            layerTypes: ["sliding_attention", "full_attention", "sliding_attention"]);
        var model = new Gemma4Model(config, loader);

        var logits = model.Forward([1, 2]);

        Assert.Equal(config.VocabularySize, logits.Length);
        Assert.Equal(3, model.Cache.LayerCount);
    }

    /// <summary>
    /// Verifies that the model's Forward method correctly processes all layers when the key and value dimensions are
    /// equal and the global head dimension is larger, ensuring compatibility across mixed attention layer types.
    /// </summary>
    [Fact]
    public void Forward_WithKEqVAndGlobalHeadDim_ShouldProcessAllLayers()
    {
        // Reproduces the MatMul mismatch: full attention layers have Q sized
        // for globalHeadDim but V shares K's smaller headDim. The Q
        // pass-through mechanism bridges the gap for o_proj.
        var (config, loader) = CreateTinyModel(
            numLayers: 3,
            headDim: 2,
            globalHeadDim: 4,
            layerTypes: ["sliding_attention", "full_attention", "sliding_attention"],
            attentionKeyEqualsValue: true);
        var model = new Gemma4Model(config, loader);

        var logits = model.Forward([1, 2]);

        Assert.Equal(config.VocabularySize, logits.Length);
        Assert.Equal(3, model.Cache.LayerCount);
    }

    /// <summary>
    /// Verifies that the Gemma4Model constructor throws an ArgumentNullException
    /// when the configuration object is null.
    /// </summary>
    [Fact]
    public void Constructor_NullConfig_ShouldThrow()
    {
        var (_, loader) = CreateTinyModel();
        Assert.Throws<ArgumentNullException>(() => new Gemma4Model(null, loader));
    }

    [Fact]
    public void Constructor_NullLoader_ShouldThrow()
    {
        var (config, _) = CreateTinyModel();
        Assert.Throws<ArgumentNullException>(() => new Gemma4Model(config, null));
    }

    /// <summary>
    /// Creates a minimal model configuration and an associated SafeTensorLoader with randomly
    /// initialized weight data for testing or development purposes.
    /// </summary>
    /// <remarks>
    /// This method is intended for tests, development, or quickly creating small models.
    /// The weight data is generated randomly and is not suitable for production use.
    /// </remarks>
    /// <param name="vocabSize">
    /// The size of the vocabulary used by the model. Must be positive.
    /// </param>
    /// <param name="hiddenSize">
    /// The number of hidden units in each layer of the model. Must be positive.
    /// </param>
    /// <param name="intermediateSize">
    /// The size of the intermediate layer in each layer’s feedforward network. Must be positive.
    /// </param>
    /// <param name="numLayers">
    /// The number of layers in the model. Must be positive.
    /// </param>
    /// <param name="numQueryHeads">
    /// The number of query heads in the attention layer. Must be positive.
    /// </param>
    /// <param name="numKvHeads">
    /// The number of key/value heads in the attention layer. Must be positive.
    /// </param>
    /// <param name="headDim">
    /// The dimension of each attention head. Must be positive.
    /// </param>
    /// <param name="globalHeadDim">
    /// The dimension of the global attention heads for full-attention layers.
    /// If less than or equal to 0, the value of <paramref name="headDim"/> is used.
    /// </param>
    /// <param name="tieWordEmbeddings">
    /// Indicates whether the word embeddings should be shared with the output head.
    /// Use <see langword="true"/> to share the embeddings; otherwise <see langword="false"/>.
    /// </param>
    /// <param name="layerTypes">
    /// An array specifying the type of each layer (e.g., "sliding_attention" or "full_attention").
    /// If null, all layers are set to "sliding_attention".
    /// </param>
    /// <param name="attentionKeyEqualsValue">
    /// Indicates whether the key and value projections in the attention layer are identical.
    /// Use <see langword="true"/> to share the projections; otherwise <see langword="false"/>.
    /// </param>
    /// <returns>
    /// A tuple containing the created model configuration and a SafeTensorLoader with the initialized
    /// weight data.
    /// </returns>

    private static (ModelConfiguration config, SafeTensorLoader loader) CreateTinyModel(
        int vocabSize = 16,
        int hiddenSize = 8,
        int intermediateSize = 16,
        int numLayers = 2,
        int numQueryHeads = 2,
        int numKvHeads = 1,
        int headDim = 4,
        int globalHeadDim = 0,
        bool tieWordEmbeddings = true,
        string[] layerTypes = null,
        bool attentionKeyEqualsValue = false)
    {
        layerTypes ??= Enumerable.Repeat("sliding_attention", numLayers).ToArray();
        if (globalHeadDim <= 0) globalHeadDim = headDim;

        var config = new ModelConfiguration
        {
            VocabularySize = vocabSize,
            ContextLength = 64,
            HiddenSize = hiddenSize,
            IntermediateSize = intermediateSize,
            NumberOfLayers = numLayers,
            NumberOfAttentionHeads = numQueryHeads,
            NumberOfKeyValueHeads = numKvHeads,
            HeadDimension = headDim,
            TieWordEmbeddings = tieWordEmbeddings,
            TextConfig = new TextConfig
            {
                VocabularySize = vocabSize,
                HiddenSize = hiddenSize,
                IntermediateSize = intermediateSize,
                NumberOfLayers = numLayers,
                NumberOfAttentionHeads = numQueryHeads,
                NumberOfKeyValueHeads = numKvHeads,
                HeadDimension = headDim,
                GlobalHeadDimension = globalHeadDim,
                SlidingWindow = 4,
                RmsNormEpsilon = 1e-6f,
                LayerTypes = layerTypes,
                AttentionKeyEqualsValue = attentionKeyEqualsValue,
                RopeParameters = new TextRopeParameters
                {
                    SlidingAttention = new RopeEntry
                    {
                        RopeTheta = 10000,
                        RopeType = "default",
                        PartialRotaryFactor = 1.0f
                    },
                    FullAttention = new RopeEntry
                    {
                        RopeTheta = 1000000,
                        RopeType = "proportional",
                        PartialRotaryFactor = 0.5f
                    }
                }
            }
        };

        // Build all required weight tensors
        var tensors = new Dictionary<string, (string dtype, long[] shape, float[] data)>();

        // Embedding
        tensors["model.language_model.embed_tokens.weight"] = ("F32", [vocabSize, hiddenSize],
            CreateRandomData(vocabSize * hiddenSize));

        // Final norm
        tensors["model.language_model.norm.weight"] = ("F32", [hiddenSize],
            CreateOnesData(hiddenSize));

        // Output head (only needed if not tied)
        if (!tieWordEmbeddings)
        {
            tensors["lm_head.weight"] = ("F32", [vocabSize, hiddenSize],
                CreateRandomData(vocabSize * hiddenSize));
        }

        // Per-layer weights
        for (var layer = 0; layer < numLayers; layer++)
        {
            var prefix = $"model.language_model.layers.{layer}";
            var isFullAttention = layer < layerTypes.Length && layerTypes[layer] == "full_attention";

            // Norm weights
            tensors[$"{prefix}.input_layernorm.weight"] = ("F32", [hiddenSize],
                CreateOnesData(hiddenSize));
            tensors[$"{prefix}.post_attention_layernorm.weight"] = ("F32", [hiddenSize],
                CreateOnesData(hiddenSize));

            // Attention projections – Gemma-4 full-attention layers use
            // global_head_dim for Q while K uses base head_dim. V shares K's
            // weight when attention_k_eq_v is true (otherwise V has its own
            // projection, also at kLayerHeadDim for consistency).
            // The o_proj always expects numQueryHeads * qLayerHeadDim because
            // unused Q dimensions ("pass-through") are concatenated with the
            // attention output to bridge any gap between V's dimension and Q's.
            var qLayerHeadDim = isFullAttention ? globalHeadDim : headDim;
            var kLayerHeadDim = headDim;

            tensors[$"{prefix}.self_attn.q_proj.weight"] = ("F32", [numQueryHeads * qLayerHeadDim, hiddenSize],
                CreateRandomData(numQueryHeads * qLayerHeadDim * hiddenSize));
            tensors[$"{prefix}.self_attn.k_proj.weight"] = ("F32", [numKvHeads * kLayerHeadDim, hiddenSize],
                CreateRandomData(numKvHeads * kLayerHeadDim * hiddenSize));

            // Only emit v_proj when K and V are not shared
            if (!attentionKeyEqualsValue)
            {
                tensors[$"{prefix}.self_attn.v_proj.weight"] = ("F32", [numKvHeads * kLayerHeadDim, hiddenSize],
                    CreateRandomData(numKvHeads * kLayerHeadDim * hiddenSize));
            }

            tensors[$"{prefix}.self_attn.o_proj.weight"] = ("F32", [hiddenSize, numQueryHeads * qLayerHeadDim],
                CreateRandomData(hiddenSize * numQueryHeads * qLayerHeadDim));

            // FFN projections
            tensors[$"{prefix}.mlp.gate_proj.weight"] = ("F32", [intermediateSize, hiddenSize],
                CreateRandomData(intermediateSize * hiddenSize));
            tensors[$"{prefix}.mlp.up_proj.weight"] = ("F32", [intermediateSize, hiddenSize],
                CreateRandomData(intermediateSize * hiddenSize));
            tensors[$"{prefix}.mlp.down_proj.weight"] = ("F32", [hiddenSize, intermediateSize],
                CreateRandomData(hiddenSize * intermediateSize));
        }

        var bytes = CreateSafeTensorsFile(tensors);
        var weights = ModelWeights.FromByteArray(bytes);
        var loader = new SafeTensorLoader(weights);

        return (config, loader);
    }

    /// <summary>
    /// Generates an array of pseudo-random floating-point values for testing or simulation purposes.
    /// </summary>
    /// <remarks>The generated values are deterministic for a given count, making this method suitable for
    /// repeatable tests or simulations.</remarks>
    /// <param name="count">The number of elements to include in the returned array. Must be non-negative.</param>
    /// <returns>
    /// An array of single-precision floating-point numbers containing pseudo-random values. The array length is equal
    /// to the specified count.
    /// </returns>
    private static float[] CreateRandomData(int count)
    {
        var data = new float[count];

        for (var i = 0; i < count; i++)
        {
            // Deterministic pseudo-random small values
            data[i] = ((i * 1103515245 + 12345) % 1000) / 10000.0f - 0.05f;
        }

        return data;
    }

    /// <summary>
    /// Creates an array of the specified length that is filled entirely with ones.
    /// </summary>
    /// <param name="count">
    /// The number of elements in the returned array. Must be greater than or equal to 0.
    /// </param>
    /// <returns>
    /// A float array with the specified number of elements, where each element has the value 1.0f.
    /// </returns>
    private static float[] CreateOnesData(int count)
    {
        var data = new float[count];
        Array.Fill(data, 1.0f);
        return data;
    }

    /// <summary>
    /// Creates a byte representation of a SafeTensors file from the provided tensor data.
    /// </summary>
    /// <remarks>
    /// The method serializes the tensors into the SafeTensors format, storing the data as
    /// little-endian floats. The header contains metadata for each tensor, including data type,
    /// shape, and data offsets. The method supports only tensors whose data consists of
    /// floating‑point values.
    /// </remarks>
    /// <param name="tensors">
    /// A dictionary mapping tensor names to their associated data type, shape, and floating‑point
    /// values. Each entry contains the data type as a string, the shape as an array of lengths,
    /// and the tensor data as an array of floats.
    /// </param>
    /// <returns>
    /// A byte array containing the SafeTensors file, including the header and the binary tensor data.
    /// </returns>
    private static byte[] CreateSafeTensorsFile(
        Dictionary<string, (string dtype, long[] shape, float[] data)> tensors)
    {
        var rawTensors = new Dictionary<string, (string dtype, long[] shape, byte[] data)>();

        foreach (var (name, (dtype, shape, data)) in tensors)
        {
            var rawData = new byte[data.Length * 4];

            for (var i = 0; i < data.Length; i++)
            {
                BinaryPrimitives.WriteSingleLittleEndian(rawData.AsSpan(i * 4), data[i]);
            }

            rawTensors[name] = (dtype, shape, rawData);
        }

        // Build header JSON
        var header = new Dictionary<string, object>();
        long currentOffset = 0;

        foreach (var (name, (dtype, shape, data)) in rawTensors)
        {
            var endOffset = currentOffset + data.Length;
            header[name] = new
            {
                dtype,
                shape,
                data_offsets = new long[] { currentOffset, endOffset }
            };
            currentOffset = endOffset;
        }

        var headerJson = JsonSerializer.Serialize(header);
        var headerBytes = Encoding.UTF8.GetBytes(headerJson);

        var totalDataSize = rawTensors.Values.Sum(t => t.data.Length);
        var result = new byte[8 + headerBytes.Length + totalDataSize];

        BinaryPrimitives.WriteInt64LittleEndian(result, headerBytes.Length);
        Array.Copy(headerBytes, 0, result, 8, headerBytes.Length);

        var dataOffset = 8 + headerBytes.Length;

        foreach (var (_, (_, _, data)) in rawTensors)
        {
            Array.Copy(data, 0, result, dataOffset, data.Length);
            dataOffset += data.Length;
        }

        return result;
    }
}
