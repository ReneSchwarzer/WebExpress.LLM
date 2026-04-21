using WebExpress.LLM.Gemma;

namespace WebExpress.LLM.Test.Gemma;

/// <summary>
/// Unit tests for <see cref="MoeBlock"/>, the Mixture-of-Experts sublayer.
/// </summary>
public sealed class UnitTestMoeBlock
{
    /// <summary>
    /// Verifies that the output tensor has the same [seqLen, hiddenSize] shape as the input.
    /// </summary>
    [Fact]
    public void Forward_ShouldProduceCorrectOutputShape()
    {
        var seqLen = 3;
        var hiddenSize = 4;
        var numExperts = 4;
        var moeInter = 2;

        var hidden = CreateInput(seqLen, hiddenSize, 0.1f);
        var router = CreateWeight(numExperts, hiddenSize, 0.2f);
        var gateUp = CreateWeight3D(numExperts, 2 * moeInter, hiddenSize, 0.05f);
        var down = CreateWeight3D(numExperts, hiddenSize, moeInter, 0.05f);

        var result = MoeBlock.Forward(hidden, router, null, null, gateUp, down, topK: 2);

        Assert.Equal(seqLen, result.Shape[0]);
        Assert.Equal(hiddenSize, result.Shape[1]);
    }

    /// <summary>
    /// Verifies that top-1 routing to a single deterministic expert reproduces the
    /// hand-computed gated-FFN output exactly.
    /// </summary>
    [Fact]
    public void Forward_TopOne_MatchesHandComputedExpertOutput()
    {
        const int hiddenSize = 2;
        const int numExperts = 2;
        const int moeInter = 2;

        // Token [1, 0] with router weights [[1,0],[0,1]] yields logits [1, 0],
        // so the top-1 selection is always expert 0 with softmax weight 1.
        var hidden = new WebExpress.LLM.Tensor.Tensor([1, hiddenSize], [1f, 0f]);

        var router = new WebExpress.LLM.Tensor.Tensor(
            [numExperts, hiddenSize],
            [1f, 0f, 0f, 1f]);

        // Expert 0: gate=I, up=I, down=I (identity-like weights).
        // Expert 1: zeros (should be ignored with top-1).
        var gateUpData = new float[numExperts * 2 * moeInter * hiddenSize];
        gateUpData[0] = 1f; gateUpData[1] = 0f;   // gate row 0
        gateUpData[2] = 0f; gateUpData[3] = 1f;   // gate row 1
        gateUpData[4] = 1f; gateUpData[5] = 0f;   // up row 0
        gateUpData[6] = 0f; gateUpData[7] = 1f;   // up row 1

        var gateUp = new WebExpress.LLM.Tensor.Tensor(
            [numExperts, 2 * moeInter, hiddenSize],
            gateUpData);

        var downData = new float[numExperts * hiddenSize * moeInter];
        downData[0] = 1f; downData[1] = 0f;       // expert 0, row 0 (hidden dim 0)
        downData[2] = 0f; downData[3] = 1f;       // expert 0, row 1 (hidden dim 1)

        var down = new WebExpress.LLM.Tensor.Tensor(
            [numExperts, hiddenSize, moeInter],
            downData);

        var result = MoeBlock.Forward(hidden, router, null, null, gateUp, down, topK: 1);

        // With hidden=[1,0] and identity-like expert 0:
        //   gate_up = [1, 0, 1, 0] -> gate=[1,0], up=[1,0]
        //   activated = GELU(gate) * up = [GELU(1), 0]
        //   output    = activated @ I = [GELU(1), 0]
        // Softmax over a single logit is 1.0, so the final weight is 1.
        var gelu1 = 0.5f * 1f * (1f + MathF.Tanh(MathF.Sqrt(2f / MathF.PI) * (1f + 0.044715f)));

        Assert.Equal(gelu1, result.Data[0], 1e-5f);
        Assert.Equal(0f, result.Data[1], 1e-6f);
    }

    /// <summary>
    /// Verifies that the MoE output is zero when all expert weights are zero,
    /// independent of the router logits.
    /// </summary>
    [Fact]
    public void Forward_ZeroExperts_ShouldReturnZero()
    {
        const int seqLen = 2;
        const int hiddenSize = 3;
        const int numExperts = 4;
        const int moeInter = 2;

        var hidden = CreateInput(seqLen, hiddenSize, 0.3f);
        var router = CreateWeight(numExperts, hiddenSize, 0.2f);

        var gateUp = new WebExpress.LLM.Tensor.Tensor(
            [numExperts, 2 * moeInter, hiddenSize],
            new float[numExperts * 2 * moeInter * hiddenSize]);

        var down = new WebExpress.LLM.Tensor.Tensor(
            [numExperts, hiddenSize, moeInter],
            new float[numExperts * hiddenSize * moeInter]);

        var result = MoeBlock.Forward(hidden, router, null, null, gateUp, down, topK: 2);

        for (var i = 0; i < result.Length; i++)
        {
            Assert.Equal(0f, result.Data[i], 1e-6f);
        }
    }

    /// <summary>
    /// Verifies that running the forward pass twice with identical inputs yields
    /// bit-for-bit identical outputs (MoE routing is deterministic).
    /// </summary>
    [Fact]
    public void Forward_ShouldBeDeterministic()
    {
        const int seqLen = 4;
        const int hiddenSize = 4;
        const int numExperts = 4;
        const int moeInter = 3;

        var hidden = CreateInput(seqLen, hiddenSize, 0.1f);
        var router = CreateWeight(numExperts, hiddenSize, 0.15f);
        var gateUp = CreateWeight3D(numExperts, 2 * moeInter, hiddenSize, 0.05f);
        var down = CreateWeight3D(numExperts, hiddenSize, moeInter, 0.05f);

        var a = MoeBlock.Forward(hidden, router, null, null, gateUp, down, topK: 2);
        var b = MoeBlock.Forward(hidden, router, null, null, gateUp, down, topK: 2);

        for (var i = 0; i < a.Length; i++)
        {
            Assert.Equal(a.Data[i], b.Data[i]);
        }
    }

    /// <summary>
    /// Verifies that per-expert scaling multiplies into the final output.
    /// </summary>
    [Fact]
    public void Forward_PerExpertScale_IsAppliedToOutput()
    {
        const int hiddenSize = 2;
        const int numExperts = 2;
        const int moeInter = 2;

        var hidden = new WebExpress.LLM.Tensor.Tensor([1, hiddenSize], [1f, 0f]);

        var router = new WebExpress.LLM.Tensor.Tensor(
            [numExperts, hiddenSize],
            [1f, 0f, 0f, 1f]);

        var gateUpData = new float[numExperts * 2 * moeInter * hiddenSize];
        gateUpData[0] = 1f; gateUpData[3] = 1f; gateUpData[4] = 1f; gateUpData[7] = 1f;

        var gateUp = new WebExpress.LLM.Tensor.Tensor(
            [numExperts, 2 * moeInter, hiddenSize], gateUpData);

        var downData = new float[numExperts * hiddenSize * moeInter];
        downData[0] = 1f; downData[3] = 1f;

        var down = new WebExpress.LLM.Tensor.Tensor(
            [numExperts, hiddenSize, moeInter], downData);

        var perExpertScale = new WebExpress.LLM.Tensor.Tensor(
            [numExperts], [0.5f, 0.5f]);

        var scaled = MoeBlock.Forward(hidden, router, null, perExpertScale, gateUp, down, topK: 1);
        var unscaled = MoeBlock.Forward(hidden, router, null, null, gateUp, down, topK: 1);

        Assert.Equal(unscaled.Data[0] * 0.5f, scaled.Data[0], 1e-5f);
    }

    /// <summary>
    /// Verifies that a null hidden input throws ArgumentNullException.
    /// </summary>
    [Fact]
    public void Forward_NullHidden_ShouldThrow()
    {
        var router = CreateWeight(2, 2, 0.1f);
        var gateUp = CreateWeight3D(2, 4, 2, 0.1f);
        var down = CreateWeight3D(2, 2, 2, 0.1f);

        Assert.Throws<ArgumentNullException>(() =>
            MoeBlock.Forward(null, router, null, null, gateUp, down, topK: 1));
    }

    /// <summary>
    /// Verifies that an invalid topK throws ArgumentOutOfRangeException.
    /// </summary>
    [Fact]
    public void Forward_InvalidTopK_ShouldThrow()
    {
        var hidden = CreateInput(1, 2, 0.1f);
        var router = CreateWeight(2, 2, 0.1f);
        var gateUp = CreateWeight3D(2, 4, 2, 0.1f);
        var down = CreateWeight3D(2, 2, 2, 0.1f);

        Assert.Throws<ArgumentOutOfRangeException>(() =>
            MoeBlock.Forward(hidden, router, null, null, gateUp, down, topK: 0));

        Assert.Throws<ArgumentOutOfRangeException>(() =>
            MoeBlock.Forward(hidden, router, null, null, gateUp, down, topK: 3));
    }

    private static WebExpress.LLM.Tensor.Tensor CreateInput(int seqLen, int hiddenSize, float scale)
    {
        var data = new float[seqLen * hiddenSize];

        for (var i = 0; i < data.Length; i++)
        {
            data[i] = ((i * 13 + 5) % 9) * scale;
        }

        return new WebExpress.LLM.Tensor.Tensor([seqLen, hiddenSize], data);
    }

    private static WebExpress.LLM.Tensor.Tensor CreateWeight(int rows, int cols, float scale)
    {
        var data = new float[rows * cols];

        for (var i = 0; i < data.Length; i++)
        {
            data[i] = ((i * 7 + 3) % 11) * scale;
        }

        return new WebExpress.LLM.Tensor.Tensor([rows, cols], data);
    }

    private static WebExpress.LLM.Tensor.Tensor CreateWeight3D(int d0, int d1, int d2, float scale)
    {
        var data = new float[d0 * d1 * d2];

        for (var i = 0; i < data.Length; i++)
        {
            data[i] = ((i * 17 + 2) % 13) * scale;
        }

        return new WebExpress.LLM.Tensor.Tensor([d0, d1, d2], data);
    }
}
