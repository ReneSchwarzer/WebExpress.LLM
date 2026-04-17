using WebExpress.LLM.Gemma;

namespace WebExpress.LLM.Test.Gemma;

public sealed class UnitTestRotaryEmbedding
{
    [Fact]
    public void Constructor_NegativeTheta_ShouldThrow()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => new RotaryEmbedding(theta: -1));
    }

    [Fact]
    public void Constructor_InvalidPartialFactor_ShouldThrow()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => new RotaryEmbedding(partialRotaryFactor: 0));
        Assert.Throws<ArgumentOutOfRangeException>(() => new RotaryEmbedding(partialRotaryFactor: 1.5f));
    }

    [Fact]
    public void Apply_2D_Position0_ShouldNotChangeVector()
    {
        // At position 0, all angles are 0, so cos=1 and sin=0 → no change
        var rope = new RotaryEmbedding(theta: 10000);
        var input = new WebExpress.LLM.Tensor.Tensor([1, 4], [1f, 2, 3, 4]);

        var result = rope.Apply(input, startPosition: 0);

        Assert.Equal(1, result.Shape[0]);
        Assert.Equal(4, result.Shape[1]);
        Assert.Equal(1.0f, result[0, 0], 1e-4f);
        Assert.Equal(2.0f, result[0, 1], 1e-4f);
        Assert.Equal(3.0f, result[0, 2], 1e-4f);
        Assert.Equal(4.0f, result[0, 3], 1e-4f);
    }

    [Fact]
    public void Apply_2D_NonZeroPosition_ShouldRotateValues()
    {
        var rope = new RotaryEmbedding(theta: 10000);
        var input = new WebExpress.LLM.Tensor.Tensor([1, 4], [1f, 0, 0, 1]);

        var result = rope.Apply(input, startPosition: 1);

        // At position 1, the first pair (i=0) should be rotated by angle = 1/theta^(0/4) = 1
        var angle = 1.0f / MathF.Pow(10000, 0.0f / 4);
        var cos0 = MathF.Cos(angle);
        var sin0 = MathF.Sin(angle);

        Assert.Equal(1.0f * cos0 - 0.0f * sin0, result[0, 0], 1e-4f);
        Assert.Equal(1.0f * sin0 + 0.0f * cos0, result[0, 1], 1e-4f);
    }

    [Fact]
    public void Apply_3D_ShouldWorkPerHead()
    {
        var rope = new RotaryEmbedding(theta: 10000);
        // 2 heads, 1 position, 4 dims per head
        var input = new WebExpress.LLM.Tensor.Tensor([2, 1, 4], [1f, 0, 0, 0, 0, 1, 0, 0]);

        var result = rope.Apply(input, startPosition: 0);

        Assert.Equal(2, result.Shape[0]);
        Assert.Equal(1, result.Shape[1]);
        Assert.Equal(4, result.Shape[2]);

        // At position 0, no rotation should occur
        Assert.Equal(1.0f, result[0, 0, 0], 1e-4f);
        Assert.Equal(0.0f, result[0, 0, 1], 1e-4f);
        Assert.Equal(0.0f, result[1, 0, 0], 1e-4f);
        Assert.Equal(1.0f, result[1, 0, 1], 1e-4f);
    }

    [Fact]
    public void Apply_PartialRotary_ShouldOnlyRotatePartOfDimension()
    {
        // partialRotaryFactor=0.5 means only first half of dims are rotated
        var rope = new RotaryEmbedding(theta: 10000, partialRotaryFactor: 0.5f);
        var input = new WebExpress.LLM.Tensor.Tensor([1, 4], [1f, 2, 3, 4]);

        var result = rope.Apply(input, startPosition: 5);

        // Last 2 dims should be untouched
        Assert.Equal(3.0f, result[0, 2], 1e-4f);
        Assert.Equal(4.0f, result[0, 3], 1e-4f);

        // First 2 dims should be rotated (values changed)
        // At position 5, first pair should be different from input
        Assert.NotEqual(1.0f, result[0, 0], 1e-2f);
    }

    [Fact]
    public void Apply_DifferentPositions_ShouldProduceDifferentResults()
    {
        var rope = new RotaryEmbedding(theta: 10000);
        var input = new WebExpress.LLM.Tensor.Tensor([1, 4], [1f, 0, 1, 0]);

        var result0 = rope.Apply(input, startPosition: 0);
        var result1 = rope.Apply(input, startPosition: 1);

        // Different positions → different rotations
        Assert.NotEqual(result0[0, 0], result1[0, 0], 1e-4f);
    }

    [Fact]
    public void Apply_1DTensor_ShouldThrow()
    {
        var rope = new RotaryEmbedding();
        var input = WebExpress.LLM.Tensor.Tensor.FromArray([1f, 2, 3, 4]);

        Assert.Throws<ArgumentException>(() => rope.Apply(input));
    }

    [Fact]
    public void Apply_SlidingAttentionConfig_ShouldUseDefaultTheta()
    {
        // Sliding attention uses theta=10000 (default)
        var rope = new RotaryEmbedding(theta: 10000, partialRotaryFactor: 1.0f);
        var input = new WebExpress.LLM.Tensor.Tensor([1, 4], [1f, 0, 0, 0]);

        var result = rope.Apply(input, startPosition: 1);

        // Verify rotation happened
        Assert.NotEqual(1.0f, result[0, 0], 1e-2f);
    }

    [Fact]
    public void Apply_FullAttentionConfig_ShouldUseLargeTheta()
    {
        // Full attention uses theta=1000000 with partial=0.25
        var rope = new RotaryEmbedding(theta: 1000000, partialRotaryFactor: 0.25f);

        // Need at least 8 dims for 25% = 2 dims (1 pair)
        var input = new WebExpress.LLM.Tensor.Tensor([1, 8], [1f, 0, 0, 0, 0, 0, 0, 0]);

        var result = rope.Apply(input, startPosition: 1);

        // With large theta, rotation angle at position 1 is small: angle = 1/theta^(0/2) = 1.0
        // cos(1) ≈ 0.54 so some rotation still occurs, but dims beyond the partial factor remain unchanged
        Assert.Equal(0.0f, result[0, 2], 1e-4f);  // Unrotated dim
        Assert.Equal(0.0f, result[0, 7], 1e-4f);  // Unrotated dim
    }
}
