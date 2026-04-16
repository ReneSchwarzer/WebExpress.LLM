using System.Buffers.Binary;
using System.Text;
using System.Text.Json;
using WebExpress.LLM.Gemma;
using WebExpress.LLM.Model;
using WebExpress.LLM.SafeTensors;
using WebExpress.LLM.Tensor;

namespace WebExpress.LLM.Test.Gemma;

public sealed class Gemma4ModelTests
{
    [Fact]
    public void Forward_ShouldProduceLogitsOfVocabSize()
    {
        var (config, loader) = CreateTinyModel();
        var model = new Gemma4Model(config, loader);

        var logits = model.Forward([1, 2, 3]);

        Assert.Equal(config.VocabularySize, logits.Length);
    }

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

    [Fact]
    public void Forward_EmptyTokens_ShouldThrow()
    {
        var (config, loader) = CreateTinyModel();
        var model = new Gemma4Model(config, loader);

        Assert.Throws<ArgumentException>(() => model.Forward(Array.Empty<int>()));
    }

    [Fact]
    public void Forward_NullTokens_ShouldThrow()
    {
        var (config, loader) = CreateTinyModel();
        var model = new Gemma4Model(config, loader);

        Assert.Throws<ArgumentNullException>(() => model.Forward(null));
    }

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

    [Fact]
    public void Forward_SingleToken_ShouldWork()
    {
        var (config, loader) = CreateTinyModel();
        var model = new Gemma4Model(config, loader);

        var logits = model.Forward([0]);

        Assert.Equal(config.VocabularySize, logits.Length);
    }

    [Fact]
    public void Forward_WithTiedEmbeddings_ShouldUseEmbedWeightForOutput()
    {
        // Create a model with tied embeddings
        var (config, loader) = CreateTinyModel(tieWordEmbeddings: true);
        var model = new Gemma4Model(config, loader);

        var logits = model.Forward([1]);

        Assert.Equal(config.VocabularySize, logits.Length);
    }

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

    // ---------------------------------------------------------------
    // Helper: Creates a minimal model configuration and SafeTensorLoader
    // ---------------------------------------------------------------
    private static (ModelConfiguration config, SafeTensorLoader loader) CreateTinyModel(
        int vocabSize = 16,
        int hiddenSize = 8,
        int intermediateSize = 16,
        int numLayers = 2,
        int numQueryHeads = 2,
        int numKvHeads = 1,
        int headDim = 4,
        bool tieWordEmbeddings = true,
        string[] layerTypes = null)
    {
        layerTypes ??= Enumerable.Repeat("sliding_attention", numLayers).ToArray();

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
                GlobalHeadDimension = headDim,
                SlidingWindow = 4,
                RmsNormEpsilon = 1e-6f,
                LayerTypes = layerTypes,
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
        tensors["model.embed_tokens.weight"] = ("F32", [vocabSize, hiddenSize],
            CreateRandomData(vocabSize * hiddenSize));

        // Final norm
        tensors["model.norm.weight"] = ("F32", [hiddenSize],
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
            var prefix = $"model.layers.{layer}";

            // Norm weights
            tensors[$"{prefix}.input_layernorm.weight"] = ("F32", [hiddenSize],
                CreateOnesData(hiddenSize));
            tensors[$"{prefix}.post_attention_layernorm.weight"] = ("F32", [hiddenSize],
                CreateOnesData(hiddenSize));

            // Attention projections
            tensors[$"{prefix}.self_attn.q_proj.weight"] = ("F32", [numQueryHeads * headDim, hiddenSize],
                CreateRandomData(numQueryHeads * headDim * hiddenSize));
            tensors[$"{prefix}.self_attn.k_proj.weight"] = ("F32", [numKvHeads * headDim, hiddenSize],
                CreateRandomData(numKvHeads * headDim * hiddenSize));
            tensors[$"{prefix}.self_attn.v_proj.weight"] = ("F32", [numKvHeads * headDim, hiddenSize],
                CreateRandomData(numKvHeads * headDim * hiddenSize));
            tensors[$"{prefix}.self_attn.o_proj.weight"] = ("F32", [hiddenSize, numQueryHeads * headDim],
                CreateRandomData(hiddenSize * numQueryHeads * headDim));

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

    private static float[] CreateOnesData(int count)
    {
        var data = new float[count];
        Array.Fill(data, 1.0f);
        return data;
    }

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
