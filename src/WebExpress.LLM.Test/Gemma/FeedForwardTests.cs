using WebExpress.LLM.Gemma;
using WebExpress.LLM.Tensor;

namespace WebExpress.LLM.Test.Gemma;

public sealed class FeedForwardTests
{
    [Fact]
    public void Forward_ShouldProduceCorrectOutputShape()
    {
        var hiddenSize = 4;
        var intermediateSize = 8;
        var seqLen = 3;

        var input = new WebExpress.LLM.Tensor.Tensor([seqLen, hiddenSize], new float[seqLen * hiddenSize]);
        var gateWeight = CreateWeight(intermediateSize, hiddenSize);
        var upWeight = CreateWeight(intermediateSize, hiddenSize);
        var downWeight = CreateWeight(hiddenSize, intermediateSize);

        var result = FeedForward.Forward(input, gateWeight, upWeight, downWeight);

        Assert.Equal(seqLen, result.Shape[0]);
        Assert.Equal(hiddenSize, result.Shape[1]);
    }

    [Fact]
    public void Forward_ZeroInput_ShouldReturnZero()
    {
        var hiddenSize = 4;
        var intermediateSize = 8;

        var input = new WebExpress.LLM.Tensor.Tensor([1, hiddenSize], new float[hiddenSize]);
        var gateWeight = CreateWeight(intermediateSize, hiddenSize);
        var upWeight = CreateWeight(intermediateSize, hiddenSize);
        var downWeight = CreateWeight(hiddenSize, intermediateSize);

        var result = FeedForward.Forward(input, gateWeight, upWeight, downWeight);

        // Zero input through linear + GELU should produce zero gate, zero up, zero output
        for (var i = 0; i < result.Length; i++)
        {
            Assert.Equal(0.0f, result.Data[i], 1e-6f);
        }
    }

    [Fact]
    public void Forward_ShouldBeDeterministic()
    {
        var hiddenSize = 4;
        var intermediateSize = 8;

        var input = CreateInput(2, hiddenSize);
        var gateWeight = CreateWeight(intermediateSize, hiddenSize);
        var upWeight = CreateWeight(intermediateSize, hiddenSize);
        var downWeight = CreateWeight(hiddenSize, intermediateSize);

        var result1 = FeedForward.Forward(input, gateWeight, upWeight, downWeight);
        var result2 = FeedForward.Forward(input, gateWeight, upWeight, downWeight);

        for (var i = 0; i < result1.Length; i++)
        {
            Assert.Equal(result1.Data[i], result2.Data[i], 1e-6f);
        }
    }

    [Fact]
    public void Forward_NullInput_ShouldThrow()
    {
        var gateWeight = CreateWeight(4, 2);
        var upWeight = CreateWeight(4, 2);
        var downWeight = CreateWeight(2, 4);

        Assert.Throws<ArgumentNullException>(() =>
            FeedForward.Forward(null, gateWeight, upWeight, downWeight));
    }

    [Fact]
    public void Forward_NullGateWeight_ShouldThrow()
    {
        var input = CreateInput(1, 2);
        var upWeight = CreateWeight(4, 2);
        var downWeight = CreateWeight(2, 4);

        Assert.Throws<ArgumentNullException>(() =>
            FeedForward.Forward(input, null, upWeight, downWeight));
    }

    [Fact]
    public void Forward_NonZeroInput_ShouldProduceNonZeroOutput()
    {
        var hiddenSize = 4;
        var intermediateSize = 8;

        // Use non-zero input
        var data = new float[hiddenSize];

        for (var i = 0; i < hiddenSize; i++)
        {
            data[i] = 1.0f;
        }

        var input = new WebExpress.LLM.Tensor.Tensor([1, hiddenSize], data);
        var gateWeight = CreateNonZeroWeight(intermediateSize, hiddenSize);
        var upWeight = CreateNonZeroWeight(intermediateSize, hiddenSize);
        var downWeight = CreateNonZeroWeight(hiddenSize, intermediateSize);

        var result = FeedForward.Forward(input, gateWeight, upWeight, downWeight);

        // At least some outputs should be non-zero
        var hasNonZero = false;

        for (var i = 0; i < result.Length; i++)
        {
            if (MathF.Abs(result.Data[i]) > 1e-6f)
            {
                hasNonZero = true;
                break;
            }
        }

        Assert.True(hasNonZero, "FFN with non-zero input should produce non-zero output.");
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

    private static WebExpress.LLM.Tensor.Tensor CreateNonZeroWeight(int rows, int cols)
    {
        var data = new float[rows * cols];

        for (var i = 0; i < data.Length; i++)
        {
            data[i] = 0.1f + ((i * 3 + 1) % 5) * 0.05f;
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
