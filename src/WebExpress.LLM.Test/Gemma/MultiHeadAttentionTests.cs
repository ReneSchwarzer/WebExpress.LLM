using WebExpress.LLM.Gemma;
using WebExpress.LLM.Tensor;

namespace WebExpress.LLM.Test.Gemma;

public sealed class MultiHeadAttentionTests
{
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

    [Fact]
    public void Forward_AsymmetricQKvHeadDim_ShouldProduceCorrectOutputShape()
    {
        // Reproduces the original bug: Q uses a larger per-head dimension
        // (globalHeadDim) than K (headDim). V uses globalHeadDim like Q,
        // so scores @ V produces the right dimension for the output projection.
        var numQueryHeads = 2;
        var numKvHeads = 1;
        var qHeadDim = 8;    // global_head_dim – larger (used by Q and V)
        var kHeadDim = 4;    // head_dim – smaller (used by K only)
        var hiddenSize = 16;
        var seqLen = 3;

        var rope = new RotaryEmbedding(theta: 10000);
        var attention = new MultiHeadAttention(
            numQueryHeads, numKvHeads, kHeadDim,
            isFullAttention: true, slidingWindowSize: 512, rope: rope);

        var input = CreateInput(seqLen, hiddenSize);

        // Q and V projections use qHeadDim, K projection uses kHeadDim
        var qWeight = CreateWeight(numQueryHeads * qHeadDim, hiddenSize);
        var kWeight = CreateWeight(numKvHeads * kHeadDim, hiddenSize);
        var vWeight = CreateWeight(numKvHeads * qHeadDim, hiddenSize);
        // Output projection matches attention output: numQueryHeads * vHeadDim
        var oWeight = CreateWeight(hiddenSize, numQueryHeads * qHeadDim);

        var result = attention.Forward(input, qWeight, kWeight, vWeight, oWeight);

        Assert.Equal(seqLen, result.Shape[0]);
        Assert.Equal(hiddenSize, result.Shape[1]);
    }

    [Fact]
    public void Forward_NullRope_ShouldThrow()
    {
        Assert.Throws<ArgumentNullException>(() =>
            new MultiHeadAttention(2, 1, 4, true, 512, null));
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
