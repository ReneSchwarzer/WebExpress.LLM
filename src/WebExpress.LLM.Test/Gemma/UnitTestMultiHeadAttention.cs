using WebExpress.LLM.Gemma;
using WebExpress.LLM.Tensor;

namespace WebExpress.LLM.Test.Gemma;

/// <summary>
/// Provides unit tests for the MultiHeadAttention component of the Gemma model.
/// </summary>
public sealed class UnitTestMultiHeadAttention
{
    /// <summary>
    /// Tests that the forward pass produces the correct output shape.
    /// </summary>
    [Fact]
    public void Forward_ShouldProduceCorrectOutputShape()
    {
        // Small test: 2 query heads, 1 KV head (GQA), headDim=4, seqLen=3, hiddenSize=8
        var numQueryHeads = 2;
        var numKvHeads = 1;
        var headDim = 4;
        var hiddenSize = numQueryHeads * headDim; // 8
        var seqLen = 3;

        var rope = new RotaryEmbedding(theta: 10000);
        var attention = new MultiHeadAttention(
            numQueryHeads, numKvHeads, headDim,
            isFullAttention: true, slidingWindowSize: 512, rope: rope);

        var input = new WebExpress.LLM.Tensor.Tensor([seqLen, hiddenSize], new float[seqLen * hiddenSize]);

        // Initialize with small random-like values
        for (var i = 0; i < input.Length; i++)
        {
            input[i] = (i % 7) * 0.1f;
        }

        // Projection weights: [projSize, hiddenSize]
        var qWeight = CreateWeight(numQueryHeads * headDim, hiddenSize);
        var kWeight = CreateWeight(numKvHeads * headDim, hiddenSize);
        var vWeight = CreateWeight(numKvHeads * headDim, hiddenSize);
        var oWeight = CreateWeight(hiddenSize, numQueryHeads * headDim);

        var result = attention.Forward(input, qWeight, kWeight, vWeight, oWeight);

        Assert.Equal(seqLen, result.Shape[0]);
        Assert.Equal(hiddenSize, result.Shape[1]);
    }

    /// <summary>
    /// Tests that the forward pass with KV cache accumulates the sequence length.
    /// </summary>
    [Fact]
    public void Forward_WithKvCache_ShouldAccumulateSequenceLength()
    {
        var numQueryHeads = 2;
        var numKvHeads = 1;
        var headDim = 4;
        var hiddenSize = numQueryHeads * headDim;

        var rope = new RotaryEmbedding(theta: 10000);
        var attention = new MultiHeadAttention(
            numQueryHeads, numKvHeads, headDim,
            isFullAttention: true, slidingWindowSize: 512, rope: rope);

        var kvCache = new KvCache();

        var qWeight = CreateWeight(numQueryHeads * headDim, hiddenSize);
        var kWeight = CreateWeight(numKvHeads * headDim, hiddenSize);
        var vWeight = CreateWeight(numKvHeads * headDim, hiddenSize);
        var oWeight = CreateWeight(hiddenSize, numQueryHeads * headDim);

        // First pass: 3 tokens
        var input1 = CreateInput(3, hiddenSize);
        attention.Forward(input1, qWeight, kWeight, vWeight, oWeight, kvCache, layerIndex: 0);

        Assert.Equal(3, kvCache.GetSequenceLength(0));

        // Second pass: 1 token
        var input2 = CreateInput(1, hiddenSize);
        attention.Forward(input2, qWeight, kWeight, vWeight, oWeight, kvCache, layerIndex: 0);

        Assert.Equal(4, kvCache.GetSequenceLength(0));
    }

    /// <summary>
    /// Tests that the forward pass with a sliding window produces output.
    /// </summary>
    [Fact]
    public void Forward_SlidingWindow_ShouldProduceOutput()
    {
        var numQueryHeads = 2;
        var numKvHeads = 2;
        var headDim = 4;
        var hiddenSize = numQueryHeads * headDim;

        var rope = new RotaryEmbedding(theta: 10000);
        var attention = new MultiHeadAttention(
            numQueryHeads, numKvHeads, headDim,
            isFullAttention: false, slidingWindowSize: 2, rope: rope);

        var input = CreateInput(5, hiddenSize);

        var qWeight = CreateWeight(numQueryHeads * headDim, hiddenSize);
        var kWeight = CreateWeight(numKvHeads * headDim, hiddenSize);
        var vWeight = CreateWeight(numKvHeads * headDim, hiddenSize);
        var oWeight = CreateWeight(hiddenSize, numQueryHeads * headDim);

        var result = attention.Forward(input, qWeight, kWeight, vWeight, oWeight);

        Assert.Equal(5, result.Shape[0]);
        Assert.Equal(hiddenSize, result.Shape[1]);
    }

    /// <summary>
    /// Tests that the forward pass is deterministic.
    /// </summary>
    [Fact]
    public void Forward_ShouldBeDeterministic()
    {
        var numQueryHeads = 2;
        var numKvHeads = 1;
        var headDim = 4;
        var hiddenSize = numQueryHeads * headDim;

        var rope = new RotaryEmbedding(theta: 10000);

        var input = CreateInput(2, hiddenSize);
        var qWeight = CreateWeight(numQueryHeads * headDim, hiddenSize);
        var kWeight = CreateWeight(numKvHeads * headDim, hiddenSize);
        var vWeight = CreateWeight(numKvHeads * headDim, hiddenSize);
        var oWeight = CreateWeight(hiddenSize, numQueryHeads * headDim);

        var attn1 = new MultiHeadAttention(numQueryHeads, numKvHeads, headDim, true, 512, rope);
        var result1 = attn1.Forward(input, qWeight, kWeight, vWeight, oWeight);

        var attn2 = new MultiHeadAttention(numQueryHeads, numKvHeads, headDim, true, 512, rope);
        var result2 = attn2.Forward(input, qWeight, kWeight, vWeight, oWeight);

        for (var i = 0; i < result1.Length; i++)
        {
            Assert.Equal(result1.Data[i], result2.Data[i], 1e-6f);
        }
    }

    /// <summary>
    /// Tests that the forward pass derives the head dimension from the weights if it is larger than the configuration.
    /// </summary>
    [Fact]
    public void Forward_WeightHeadDimLargerThanConfig_ShouldDeriveFromWeights()
    {
        // Simulates a mismatch where headDim in the config (2) is smaller
        // than the actual projection weight dimension (4 per head).
        // Before the fix this would throw an ArrayCopy out-of-bounds error.
        var numQueryHeads = 2;
        var numKvHeads = 1;
        var actualHeadDim = 4;
        var configHeadDim = 2; // intentionally wrong / smaller
        var hiddenSize = 8;
        var seqLen = 3;

        var rope = new RotaryEmbedding(theta: 10000);
        var attention = new MultiHeadAttention(
            numQueryHeads, numKvHeads, configHeadDim,
            isFullAttention: true, slidingWindowSize: 512, rope: rope);

        var input = CreateInput(seqLen, hiddenSize);

        // Weights use actualHeadDim, not configHeadDim
        var qWeight = CreateWeight(numQueryHeads * actualHeadDim, hiddenSize);
        var kWeight = CreateWeight(numKvHeads * actualHeadDim, hiddenSize);
        var vWeight = CreateWeight(numKvHeads * actualHeadDim, hiddenSize);
        var oWeight = CreateWeight(hiddenSize, numQueryHeads * actualHeadDim);

        var result = attention.Forward(input, qWeight, kWeight, vWeight, oWeight);

        Assert.Equal(seqLen, result.Shape[0]);
        Assert.Equal(hiddenSize, result.Shape[1]);
    }

    /// <summary>
    /// Tests that the forward pass produces the correct output shape with asymmetric Q and KV head dimensions.
    /// </summary>
    [Fact]
    public void Forward_AsymmetricQKvHeadDim_ShouldProduceCorrectOutputShape()
    {
        // Reproduces the original bug: Q uses a larger per-head dimension
        // (globalHeadDim) than K (headDim). V uses kHeadDim (same as K)
        // to simulate the attention_k_eq_v case. The Q pass-through
        // mechanism bridges the gap so o_proj gets the expected dimension.
        var numQueryHeads = 2;
        var numKvHeads = 1;
        var qHeadDim = 8;    // global_head_dim – larger (used by Q)
        var kHeadDim = 4;    // head_dim – smaller (used by K and V)
        var hiddenSize = 16;
        var seqLen = 3;

        var rope = new RotaryEmbedding(theta: 10000);
        var attention = new MultiHeadAttention(
            numQueryHeads, numKvHeads, kHeadDim,
            isFullAttention: true, slidingWindowSize: 512, rope: rope);

        var input = CreateInput(seqLen, hiddenSize);

        // Q projection uses qHeadDim; K and V projections use kHeadDim
        var qWeight = CreateWeight(numQueryHeads * qHeadDim, hiddenSize);
        var kWeight = CreateWeight(numKvHeads * kHeadDim, hiddenSize);
        var vWeight = CreateWeight(numKvHeads * kHeadDim, hiddenSize);
        // Output projection expects numQueryHeads * qHeadDim (pass-through fills the gap)
        var oWeight = CreateWeight(hiddenSize, numQueryHeads * qHeadDim);

        var result = attention.Forward(input, qWeight, kWeight, vWeight, oWeight);

        Assert.Equal(seqLen, result.Shape[0]);
        Assert.Equal(hiddenSize, result.Shape[1]);
    }

    /// <summary>
    /// Tests that the constructor throws an exception when the rotary embedding is null.
    /// </summary>
    [Fact]
    public void Forward_NullRope_ShouldThrow()
    {
        Assert.Throws<ArgumentNullException>(() =>
            new MultiHeadAttention(2, 1, 4, true, 512, null));
    }

    /// <summary>
    /// Tests that passing q_norm and k_norm weights of all ones leaves Q and K
    /// unit-normalised over head_dim (i.e. the attention output changes compared
    /// to the null-norm baseline, but stays finite and deterministic).
    /// </summary>
    [Fact]
    public void Forward_WithQkNorm_ShouldAffectOutputDeterministically()
    {
        var numQueryHeads = 2;
        var numKvHeads = 1;
        var headDim = 4;
        var hiddenSize = numQueryHeads * headDim;
        var seqLen = 3;

        var rope = new RotaryEmbedding(theta: 10000);
        var attention = new MultiHeadAttention(
            numQueryHeads, numKvHeads, headDim,
            isFullAttention: true, slidingWindowSize: 512, rope: rope);

        var input = CreateInput(seqLen, hiddenSize);

        var qWeight = CreateWeight(numQueryHeads * headDim, hiddenSize);
        var kWeight = CreateWeight(numKvHeads * headDim, hiddenSize);
        var vWeight = CreateWeight(numKvHeads * headDim, hiddenSize);
        var oWeight = CreateWeight(hiddenSize, numQueryHeads * headDim);

        // q_norm / k_norm of shape [headDim]; values are all ones so that
        // RmsNorm acts as pure L2-normalisation along the head dimension.
        var qNormData = new float[headDim];
        var kNormData = new float[headDim];

        for (var i = 0; i < headDim; i++)
        {
            qNormData[i] = 1f;
            kNormData[i] = 1f;
        }

        var qNorm = new WebExpress.LLM.Tensor.Tensor([headDim], qNormData);
        var kNorm = new WebExpress.LLM.Tensor.Tensor([headDim], kNormData);

        var withoutNorm = attention.Forward(input, qWeight, kWeight, vWeight, oWeight);
        var withNorm = attention.Forward(
            input, qWeight, kWeight, vWeight, oWeight,
            kvCache: null, layerIndex: 0,
            qNormWeight: qNorm, kNormWeight: kNorm, rmsNormEpsilon: 1e-6f);

        // Shapes must match
        Assert.Equal(withoutNorm.Shape[0], withNorm.Shape[0]);
        Assert.Equal(withoutNorm.Shape[1], withNorm.Shape[1]);

        // Output must change when norms are applied
        var anyDifference = false;

        for (var i = 0; i < withoutNorm.Length; i++)
        {
            Assert.False(float.IsNaN(withNorm.Data[i]));
            Assert.False(float.IsInfinity(withNorm.Data[i]));

            if (MathF.Abs(withoutNorm.Data[i] - withNorm.Data[i]) > 1e-5f)
            {
                anyDifference = true;
            }
        }

        Assert.True(anyDifference);

        // Determinism: same inputs + same norms yield identical outputs
        var withNorm2 = attention.Forward(
            input, qWeight, kWeight, vWeight, oWeight,
            kvCache: null, layerIndex: 0,
            qNormWeight: qNorm, kNormWeight: kNorm, rmsNormEpsilon: 1e-6f);

        for (var i = 0; i < withNorm.Length; i++)
        {
            Assert.Equal(withNorm.Data[i], withNorm2.Data[i], 1e-6f);
        }
    }

    /// <summary>
    /// Tests that enabling an attention-logits soft cap produces a different
    /// (bounded) output compared to the uncapped baseline, and that the result
    /// remains deterministic.
    /// </summary>
    [Fact]
    public void Forward_WithAttentionLogitsSoftcap_ShouldBoundScoresAndStayDeterministic()
    {
        var numQueryHeads = 2;
        var numKvHeads = 1;
        var headDim = 4;
        var hiddenSize = numQueryHeads * headDim;
        var seqLen = 4;

        var rope = new RotaryEmbedding(theta: 10000);

        // Large weights so raw scores exceed the cap significantly.
        var qWeight = CreateLargeWeight(numQueryHeads * headDim, hiddenSize);
        var kWeight = CreateLargeWeight(numKvHeads * headDim, hiddenSize);
        var vWeight = CreateLargeWeight(numKvHeads * headDim, hiddenSize);
        var oWeight = CreateLargeWeight(hiddenSize, numQueryHeads * headDim);
        var input = CreateLargeInput(seqLen, hiddenSize);

        var baseline = new MultiHeadAttention(
            numQueryHeads, numKvHeads, headDim,
            isFullAttention: true, slidingWindowSize: 512, rope: rope,
            attentionLogitsSoftcap: 0f);
        var baselineOut = baseline.Forward(input, qWeight, kWeight, vWeight, oWeight);

        var capped = new MultiHeadAttention(
            numQueryHeads, numKvHeads, headDim,
            isFullAttention: true, slidingWindowSize: 512, rope: rope,
            attentionLogitsSoftcap: 1.0f);
        var cappedOut = capped.Forward(input, qWeight, kWeight, vWeight, oWeight);

        // Shapes match
        Assert.Equal(baselineOut.Shape[0], cappedOut.Shape[0]);
        Assert.Equal(baselineOut.Shape[1], cappedOut.Shape[1]);

        // Soft cap must change the output (scores pre-softmax were clearly > 1)
        var anyDifference = false;

        for (var i = 0; i < baselineOut.Length; i++)
        {
            Assert.False(float.IsNaN(cappedOut.Data[i]));
            Assert.False(float.IsInfinity(cappedOut.Data[i]));

            if (MathF.Abs(baselineOut.Data[i] - cappedOut.Data[i]) > 1e-5f)
            {
                anyDifference = true;
            }
        }

        Assert.True(anyDifference);

        // Determinism: rerun with the same cap produces identical output.
        var capped2 = new MultiHeadAttention(
            numQueryHeads, numKvHeads, headDim,
            isFullAttention: true, slidingWindowSize: 512, rope: rope,
            attentionLogitsSoftcap: 1.0f);
        var cappedOut2 = capped2.Forward(input, qWeight, kWeight, vWeight, oWeight);

        for (var i = 0; i < cappedOut.Length; i++)
        {
            Assert.Equal(cappedOut.Data[i], cappedOut2.Data[i], 1e-6f);
        }
    }

    private static WebExpress.LLM.Tensor.Tensor CreateLargeWeight(int rows, int cols)
    {
        var data = new float[rows * cols];

        for (var i = 0; i < data.Length; i++)
        {
            data[i] = ((i * 7 + 3) % 11 - 5) * 0.5f;
        }

        return new WebExpress.LLM.Tensor.Tensor([rows, cols], data);
    }

    private static WebExpress.LLM.Tensor.Tensor CreateLargeInput(int seqLen, int hiddenSize)
    {
        var data = new float[seqLen * hiddenSize];

        for (var i = 0; i < data.Length; i++)
        {
            data[i] = ((i * 13 + 5) % 9 - 4) * 1.0f;
        }

        return new WebExpress.LLM.Tensor.Tensor([seqLen, hiddenSize], data);
    }

    private static WebExpress.LLM.Tensor.Tensor CreateWeight(int rows, int cols)
    {
        var data = new float[rows * cols];

        for (var i = 0; i < data.Length; i++)
        {
            data[i] = ((i * 7 + 3) % 11) * 0.01f;
        }

        return new WebExpress.LLM.Tensor.Tensor([rows, cols], data);
    }

    private static WebExpress.LLM.Tensor.Tensor CreateInput(int seqLen, int hiddenSize)
    {
        var data = new float[seqLen * hiddenSize];

        for (var i = 0; i < data.Length; i++)
        {
            data[i] = ((i * 13 + 5) % 9) * 0.1f;
        }

        return new WebExpress.LLM.Tensor.Tensor([seqLen, hiddenSize], data);
    }
}
