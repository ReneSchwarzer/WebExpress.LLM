using WebExpress.LLM.Tensor;

namespace WebExpress.LLM.Test.Tensor;

public sealed class TensorOperationsTests
{
    [Fact]
    public void MatMul_ShouldComputeCorrectProduct()
    {
        // [2,3] x [3,2] = [2,2]
        var a = new WebExpress.LLM.Tensor.Tensor([2, 3], [1f, 2, 3, 4, 5, 6]);
        var b = new WebExpress.LLM.Tensor.Tensor([3, 2], [7f, 8, 9, 10, 11, 12]);

        var result = TensorOperations.MatMul(a, b);

        Assert.Equal(2, result.Shape[0]);
        Assert.Equal(2, result.Shape[1]);
        // Row 0: 1*7+2*9+3*11=58, 1*8+2*10+3*12=64
        Assert.Equal(58.0f, result[0, 0]);
        Assert.Equal(64.0f, result[0, 1]);
        // Row 1: 4*7+5*9+6*11=139, 4*8+5*10+6*12=154
        Assert.Equal(139.0f, result[1, 0]);
        Assert.Equal(154.0f, result[1, 1]);
    }

    [Fact]
    public void MatMul_InnerDimensionMismatch_ShouldThrow()
    {
        var a = new WebExpress.LLM.Tensor.Tensor([2, 3], new float[6]);
        var b = new WebExpress.LLM.Tensor.Tensor([2, 2], new float[4]);

        Assert.Throws<ArgumentException>(() => TensorOperations.MatMul(a, b));
    }

    [Fact]
    public void MatMul_Non2DTensor_ShouldThrow()
    {
        var a = new WebExpress.LLM.Tensor.Tensor(3);
        var b = new WebExpress.LLM.Tensor.Tensor([3, 2], new float[6]);

        Assert.Throws<ArgumentException>(() => TensorOperations.MatMul(a, b));
    }

    [Fact]
    public void MatMul_IdentityMatrix_ShouldReturnSame()
    {
        var a = new WebExpress.LLM.Tensor.Tensor([2, 3], [1f, 2, 3, 4, 5, 6]);
        var identity = new WebExpress.LLM.Tensor.Tensor([3, 3], [1f, 0, 0, 0, 1, 0, 0, 0, 1]);

        var result = TensorOperations.MatMul(a, identity);

        Assert.Equal(2, result.Shape[0]);
        Assert.Equal(3, result.Shape[1]);

        for (var i = 0; i < 6; i++)
        {
            Assert.Equal(a.Data[i], result.Data[i], 1e-6f);
        }
    }

    [Fact]
    public void BatchMatMul_ShouldComputePerBatch()
    {
        // batch=2, [2,2] x [2,2]
        var a = new WebExpress.LLM.Tensor.Tensor([2, 2, 2], [1f, 0, 0, 1, 2, 0, 0, 2]);
        var b = new WebExpress.LLM.Tensor.Tensor([2, 2, 2], [3f, 4, 5, 6, 7, 8, 9, 10]);

        var result = TensorOperations.BatchMatMul(a, b);

        Assert.Equal(2, result.Shape[0]);
        Assert.Equal(2, result.Shape[1]);
        Assert.Equal(2, result.Shape[2]);

        // Batch 0: Identity × B = B
        Assert.Equal(3.0f, result[0, 0, 0]);
        Assert.Equal(4.0f, result[0, 0, 1]);
        Assert.Equal(5.0f, result[0, 1, 0]);
        Assert.Equal(6.0f, result[0, 1, 1]);

        // Batch 1: 2*I × B = 2*B
        Assert.Equal(14.0f, result[1, 0, 0]);
        Assert.Equal(16.0f, result[1, 0, 1]);
        Assert.Equal(18.0f, result[1, 1, 0]);
        Assert.Equal(20.0f, result[1, 1, 1]);
    }

    [Fact]
    public void Softmax_ShouldNormalizeToDistribution()
    {
        var input = WebExpress.LLM.Tensor.Tensor.FromArray([1.0f, 2.0f, 3.0f]);
        var result = TensorOperations.Softmax(input);

        Assert.Equal(3, result.Length);

        // Sum should be 1.0
        var sum = result[0] + result[1] + result[2];
        Assert.Equal(1.0f, sum, 1e-5f);

        // Values should be monotonically increasing
        Assert.True(result[0] < result[1]);
        Assert.True(result[1] < result[2]);
    }

    [Fact]
    public void Softmax_2D_ShouldNormalizePerRow()
    {
        var input = new WebExpress.LLM.Tensor.Tensor([2, 3], [1f, 2, 3, 4, 5, 6]);
        var result = TensorOperations.Softmax(input);

        // Each row should sum to 1
        var row0Sum = result[0, 0] + result[0, 1] + result[0, 2];
        var row1Sum = result[1, 0] + result[1, 1] + result[1, 2];

        Assert.Equal(1.0f, row0Sum, 1e-5f);
        Assert.Equal(1.0f, row1Sum, 1e-5f);
    }

    [Fact]
    public void Softmax_UniformInput_ShouldReturnUniform()
    {
        var input = WebExpress.LLM.Tensor.Tensor.FromArray([5.0f, 5.0f, 5.0f]);
        var result = TensorOperations.Softmax(input);

        Assert.Equal(1.0f / 3, result[0], 1e-5f);
        Assert.Equal(1.0f / 3, result[1], 1e-5f);
        Assert.Equal(1.0f / 3, result[2], 1e-5f);
    }

    [Fact]
    public void Gelu_ShouldApplyActivation()
    {
        var input = WebExpress.LLM.Tensor.Tensor.FromArray([0.0f, 1.0f, -1.0f, 2.0f]);
        var result = TensorOperations.Gelu(input);

        // GELU(0) ≈ 0
        Assert.Equal(0.0f, result[0], 1e-4f);
        // GELU(1) ≈ 0.841
        Assert.InRange(result[1], 0.83f, 0.85f);
        // GELU(-1) ≈ -0.159
        Assert.InRange(result[2], -0.17f, -0.15f);
        // GELU(2) ≈ 1.955
        Assert.InRange(result[3], 1.94f, 1.97f);
    }

    [Fact]
    public void RmsNorm_ShouldNormalizeCorrectly()
    {
        var input = new WebExpress.LLM.Tensor.Tensor([1, 4], [2f, 4, 6, 8]);
        var weight = WebExpress.LLM.Tensor.Tensor.FromArray([1f, 1, 1, 1]);

        var result = TensorOperations.RmsNorm(input, weight, epsilon: 0);

        // RMS = sqrt((4+16+36+64)/4) = sqrt(30) ≈ 5.477
        var rms = MathF.Sqrt((4 + 16 + 36 + 64) / 4.0f);

        Assert.Equal(2.0f / rms, result[0, 0], 1e-4f);
        Assert.Equal(4.0f / rms, result[0, 1], 1e-4f);
        Assert.Equal(6.0f / rms, result[0, 2], 1e-4f);
        Assert.Equal(8.0f / rms, result[0, 3], 1e-4f);
    }

    [Fact]
    public void RmsNorm_WithWeight_ShouldScaleResult()
    {
        var input = new WebExpress.LLM.Tensor.Tensor([1, 2], [3f, 4]);
        var weight = WebExpress.LLM.Tensor.Tensor.FromArray([2f, 0.5f]);

        var result = TensorOperations.RmsNorm(input, weight, epsilon: 0);

        var rms = MathF.Sqrt((9 + 16) / 2.0f);
        Assert.Equal(3.0f / rms * 2.0f, result[0, 0], 1e-4f);
        Assert.Equal(4.0f / rms * 0.5f, result[0, 1], 1e-4f);
    }

    [Fact]
    public void RmsNorm_WeightDimensionMismatch_ShouldThrow()
    {
        var input = new WebExpress.LLM.Tensor.Tensor([1, 4], new float[4]);
        var weight = WebExpress.LLM.Tensor.Tensor.FromArray([1f, 1, 1]);

        Assert.Throws<ArgumentException>(() => TensorOperations.RmsNorm(input, weight));
    }

    [Fact]
    public void Tanh_ShouldApplyElementWise()
    {
        var input = WebExpress.LLM.Tensor.Tensor.FromArray([0.0f, 1.0f, -1.0f]);
        var result = TensorOperations.Tanh(input);

        Assert.Equal(0.0f, result[0], 1e-6f);
        Assert.Equal(MathF.Tanh(1.0f), result[1], 1e-6f);
        Assert.Equal(MathF.Tanh(-1.0f), result[2], 1e-6f);
    }

    [Fact]
    public void Sqrt_ShouldComputeElementWise()
    {
        var input = WebExpress.LLM.Tensor.Tensor.FromArray([4.0f, 9.0f, 16.0f]);
        var result = TensorOperations.Sqrt(input);

        Assert.Equal(2.0f, result[0], 1e-6f);
        Assert.Equal(3.0f, result[1], 1e-6f);
        Assert.Equal(4.0f, result[2], 1e-6f);
    }

    [Fact]
    public void Silu_ShouldApplyActivation()
    {
        var input = WebExpress.LLM.Tensor.Tensor.FromArray([0.0f, 1.0f, -1.0f]);
        var result = TensorOperations.Silu(input);

        // SiLU(0) = 0 * sigmoid(0) = 0
        Assert.Equal(0.0f, result[0], 1e-4f);
        // SiLU(1) = 1 * sigmoid(1) ≈ 0.731
        Assert.InRange(result[1], 0.72f, 0.74f);
        // SiLU(-1) = -1 * sigmoid(-1) ≈ -0.269
        Assert.InRange(result[2], -0.28f, -0.26f);
    }

    [Fact]
    public void CausalMask_ShouldMaskFuture()
    {
        var mask = TensorOperations.CausalMask(3);

        Assert.Equal(3, mask.Shape[0]);
        Assert.Equal(3, mask.Shape[1]);

        // Diagonal and below should be 0
        Assert.Equal(0.0f, mask[0, 0]);
        Assert.Equal(0.0f, mask[1, 0]);
        Assert.Equal(0.0f, mask[1, 1]);
        Assert.Equal(0.0f, mask[2, 0]);
        Assert.Equal(0.0f, mask[2, 1]);
        Assert.Equal(0.0f, mask[2, 2]);

        // Above diagonal should be -inf
        Assert.Equal(float.NegativeInfinity, mask[0, 1]);
        Assert.Equal(float.NegativeInfinity, mask[0, 2]);
        Assert.Equal(float.NegativeInfinity, mask[1, 2]);
    }

    [Fact]
    public void SlidingWindowMask_ShouldLimitAttentionRange()
    {
        var mask = TensorOperations.SlidingWindowMask(5, windowSize: 2);

        // Position 0 can see: [0]
        Assert.Equal(0.0f, mask[0, 0]);
        Assert.Equal(float.NegativeInfinity, mask[0, 1]);

        // Position 1 can see: [0, 1]
        Assert.Equal(0.0f, mask[1, 0]);
        Assert.Equal(0.0f, mask[1, 1]);
        Assert.Equal(float.NegativeInfinity, mask[1, 2]);

        // Position 2 can see: [1, 2] (not 0, since window=2)
        Assert.Equal(float.NegativeInfinity, mask[2, 0]);
        Assert.Equal(0.0f, mask[2, 1]);
        Assert.Equal(0.0f, mask[2, 2]);

        // Position 4 can see: [3, 4]
        Assert.Equal(float.NegativeInfinity, mask[4, 2]);
        Assert.Equal(0.0f, mask[4, 3]);
        Assert.Equal(0.0f, mask[4, 4]);
    }

    [Fact]
    public void Concatenate_ShouldJoinAlongDim0()
    {
        var a = new WebExpress.LLM.Tensor.Tensor([2, 3], [1f, 2, 3, 4, 5, 6]);
        var b = new WebExpress.LLM.Tensor.Tensor([1, 3], [7f, 8, 9]);

        var result = TensorOperations.Concatenate(a, b, dim: 0);

        Assert.Equal(3, result.Shape[0]);
        Assert.Equal(3, result.Shape[1]);
        Assert.Equal(1.0f, result[0, 0]);
        Assert.Equal(7.0f, result[2, 0]);
        Assert.Equal(9.0f, result[2, 2]);
    }

    [Fact]
    public void Concatenate_ShouldJoinAlongDim1()
    {
        var a = new WebExpress.LLM.Tensor.Tensor([2, 2], [1f, 2, 3, 4]);
        var b = new WebExpress.LLM.Tensor.Tensor([2, 1], [5f, 6]);

        var result = TensorOperations.Concatenate(a, b, dim: 1);

        Assert.Equal(2, result.Shape[0]);
        Assert.Equal(3, result.Shape[1]);
        Assert.Equal(1.0f, result[0, 0]);
        Assert.Equal(2.0f, result[0, 1]);
        Assert.Equal(5.0f, result[0, 2]);
        Assert.Equal(3.0f, result[1, 0]);
        Assert.Equal(4.0f, result[1, 1]);
        Assert.Equal(6.0f, result[1, 2]);
    }

    [Fact]
    public void Concatenate_DimensionMismatch_ShouldThrow()
    {
        var a = new WebExpress.LLM.Tensor.Tensor([2, 3], new float[6]);
        var b = new WebExpress.LLM.Tensor.Tensor([2, 4], new float[8]);

        Assert.Throws<ArgumentException>(() => TensorOperations.Concatenate(a, b, dim: 0));
    }

    [Fact]
    public void EmbeddingLookup_ShouldSelectRows()
    {
        // Embedding matrix: 4 tokens, 3 dims each
        var embed = new WebExpress.LLM.Tensor.Tensor([4, 3], [
            0.1f, 0.2f, 0.3f,  // token 0
            0.4f, 0.5f, 0.6f,  // token 1
            0.7f, 0.8f, 0.9f,  // token 2
            1.0f, 1.1f, 1.2f   // token 3
        ]);

        var result = TensorOperations.EmbeddingLookup(embed, [2, 0, 3]);

        Assert.Equal(3, result.Shape[0]); // 3 tokens
        Assert.Equal(3, result.Shape[1]); // 3 dims

        // Token 2
        Assert.Equal(0.7f, result[0, 0]);
        Assert.Equal(0.8f, result[0, 1]);
        Assert.Equal(0.9f, result[0, 2]);

        // Token 0
        Assert.Equal(0.1f, result[1, 0]);

        // Token 3
        Assert.Equal(1.0f, result[2, 0]);
    }

    [Fact]
    public void EmbeddingLookup_Non2DWeight_ShouldThrow()
    {
        var weight = WebExpress.LLM.Tensor.Tensor.FromArray([1f, 2, 3]);
        Assert.Throws<ArgumentException>(() => TensorOperations.EmbeddingLookup(weight, [0]));
    }
}
