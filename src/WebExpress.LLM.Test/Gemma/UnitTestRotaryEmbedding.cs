using WebExpress.LLM.Gemma;

namespace WebExpress.LLM.Test.Gemma;

/// <summary>
/// Provides unit tests for the RotaryEmbedding component, ensuring correct application of rotary positional embeddings.
/// </summary>
public sealed class UnitTestRotaryEmbedding
{
    /// <summary>
    /// Tests that the constructor throws an exception when the theta value is negative.
    /// </summary>
    [Fact]
    public void Constructor_NegativeTheta_ShouldThrow()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => new RotaryEmbedding(theta: -1));
    }

    /// <summary>
    /// Tests that the constructor throws an exception when the partial rotary factor is invalid.
    /// </summary>
    [Fact]
    public void Constructor_InvalidPartialFactor_ShouldThrow()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => new RotaryEmbedding(partialRotaryFactor: 0));
        Assert.Throws<ArgumentOutOfRangeException>(() => new RotaryEmbedding(partialRotaryFactor: 1.5f));
    }

    /// <summary>
    /// Tests that applying the rotary embedding at position 0 does not change the vector.
    /// </summary>
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

    /// <summary>
    /// Tests that applying the rotary embedding at a non-zero position rotates
    /// the values using the split-halves NeoX-style pattern (rotation pairs
    /// are <c>(x_j, x_{j + H/2})</c>, not consecutive pairs).
    /// </summary>
    [Fact]
    public void Apply_2D_NonZeroPosition_ShouldRotateValues()
    {
        var rope = new RotaryEmbedding(theta: 10000);
        var input = new WebExpress.LLM.Tensor.Tensor([1, 4], [1f, 0, 0, 1]);

        var result = rope.Apply(input, startPosition: 1);

        // headDim=4, half=2, ropeAngles=2 → rotates pairs (j=0,2) and (j=1,3).
        //   j=0: x0=input[0,0]=1, x1=input[0,2]=0
        //        result[0,0] = x0*cos(angle0) - x1*sin(angle0) = cos(angle0)
        //        result[0,2] = x1*cos(angle0) + x0*sin(angle0) = sin(angle0)
        //   j=1: x0=input[0,1]=0, x1=input[0,3]=1
        //        result[0,1] = x0*cos(angle1) - x1*sin(angle1) = -sin(angle1)
        //        result[0,3] = x1*cos(angle1) + x0*sin(angle1) = cos(angle1)
        var freq0 = 1.0f / MathF.Pow(10000, 2.0f * 0 / 4);
        var freq1 = 1.0f / MathF.Pow(10000, 2.0f * 1 / 4);
        var angle0 = 1.0f * freq0;
        var angle1 = 1.0f * freq1;

        Assert.Equal(MathF.Cos(angle0), result[0, 0], 1e-4f);
        Assert.Equal(-MathF.Sin(angle1), result[0, 1], 1e-4f);
        Assert.Equal(MathF.Sin(angle0), result[0, 2], 1e-4f);
        Assert.Equal(MathF.Cos(angle1), result[0, 3], 1e-4f);
    }

    /// <summary>
    /// Tests that applying the rotary embedding to a 3D tensor works per head.
    /// </summary>
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

    /// <summary>
    /// Tests that applying a partial rotary embedding only rotates the angle
    /// pairs covered by the partial factor. Uses split-halves geometry, so
    /// for headDim=4 with partial=0.5 only the (j=0, j=2) pair rotates and
    /// indices 1 and 3 stay untouched.
    /// </summary>
    [Fact]
    public void Apply_PartialRotary_ShouldOnlyRotatePartOfDimension()
    {
        var rope = new RotaryEmbedding(theta: 10000, partialRotaryFactor: 0.5f);
        var input = new WebExpress.LLM.Tensor.Tensor([1, 4], [1f, 2, 3, 4]);

        var result = rope.Apply(input, startPosition: 5);

        // ropeAngles = (int)(0.5 * 4 / 2) = 1 → only j=0 pair rotates.
        // Indices 1 (= j=1 in first half, beyond ropeAngles) and 3 (= j=1 in
        // second half) stay at their original values.
        Assert.Equal(2.0f, result[0, 1], 1e-4f);
        Assert.Equal(4.0f, result[0, 3], 1e-4f);

        // The (j=0) pair (index 0 ↔ index 2) is rotated by angle = 5.
        var angle0 = 5.0f * (1.0f / MathF.Pow(10000, 2.0f * 0 / 4));
        var cos0 = MathF.Cos(angle0);
        var sin0 = MathF.Sin(angle0);

        Assert.Equal(1.0f * cos0 - 3.0f * sin0, result[0, 0], 1e-4f);
        Assert.Equal(3.0f * cos0 + 1.0f * sin0, result[0, 2], 1e-4f);
    }

    /// <summary>
    /// Tests that applying the rotary embedding at different positions produces different results.
    /// </summary>
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

    /// <summary>
    /// Tests that applying the rotary embedding to a 1D tensor throws an exception.
    /// </summary>
    [Fact]
    public void Apply_1DTensor_ShouldThrow()
    {
        var rope = new RotaryEmbedding();
        var input = WebExpress.LLM.Tensor.Tensor.FromArray([1f, 2, 3, 4]);

        Assert.Throws<ArgumentException>(() => rope.Apply(input));
    }

    /// <summary>
    /// Tests that the sliding attention configuration uses the default theta value.
    /// </summary>
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

    /// <summary>
    /// Tests that the full attention configuration uses a large theta value.
    /// </summary>
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
