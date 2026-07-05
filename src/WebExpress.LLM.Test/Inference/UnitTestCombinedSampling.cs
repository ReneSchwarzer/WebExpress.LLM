using WebExpress.LLM.Inference;

namespace WebExpress.LLM.Test.Inference;

/// <summary>
/// Provides unit tests for the <see cref="CombinedSampling"/> strategy, covering the combined
/// temperature, top-k, and top-p pipeline as well as its constructor validation.
/// </summary>
public sealed class UnitTestCombinedSampling
{
    private static readonly float[] Logits = [0.1f, 0.5f, 0.3f, 0.8f, 0.2f];

    /// <summary>
    /// Tests that sampling with a fixed seed is deterministic.
    /// </summary>
    [Fact]
    public void Sample_WithSeed_ShouldBeDeterministic()
    {
        var sampler1 = new CombinedSampling(temperature: 1.0f, topK: 3, topP: 0.9f, seed: 42);
        var sampler2 = new CombinedSampling(temperature: 1.0f, topK: 3, topP: 0.9f, seed: 42);

        Assert.Equal(sampler1.Sample(Logits), sampler2.Sample(Logits));
    }

    /// <summary>
    /// Tests that a temperature of zero degenerates to greedy (argmax) decoding.
    /// </summary>
    [Fact]
    public void Sample_WithZeroTemperature_ShouldSelectArgMax()
    {
        var sampler = new CombinedSampling(temperature: 0.0f, topK: null, topP: null, seed: 1);

        // The highest logit (0.8) is at index 3.
        for (var i = 0; i < 20; i++)
        {
            Assert.Equal(3, sampler.Sample(Logits));
        }
    }

    /// <summary>
    /// Tests that combining top-k and top-p only ever selects tokens within the top-k set.
    /// </summary>
    [Fact]
    public void Sample_WithTopKAndTopP_ShouldSelectFromTopKTokens()
    {
        var sampler = new CombinedSampling(temperature: 1.0f, topK: 3, topP: 0.95f, seed: 42);

        var results = new HashSet<int>();
        for (var i = 0; i < 200; i++)
        {
            results.Add(sampler.Sample(Logits));
        }

        // The three highest logits are at indices 3 (0.8), 1 (0.5), and 2 (0.3).
        var topKIndices = new[] { 1, 2, 3 };
        Assert.True(results.All(result => topKIndices.Contains(result)));
    }

    /// <summary>
    /// Tests that a very small top-p restricts the selection to the single most probable token.
    /// </summary>
    [Fact]
    public void Sample_WithSmallTopP_ShouldSelectMostProbableToken()
    {
        var sampler = new CombinedSampling(temperature: 1.0f, topK: null, topP: 0.01f, seed: 7);

        for (var i = 0; i < 20; i++)
        {
            Assert.Equal(3, sampler.Sample(Logits));
        }
    }

    /// <summary>
    /// Tests that a top-k of one always selects the most probable token regardless of top-p.
    /// </summary>
    [Fact]
    public void Sample_WithTopKOfOne_ShouldSelectMostProbableToken()
    {
        var sampler = new CombinedSampling(temperature: 1.0f, topK: 1, topP: 0.9f, seed: 3);

        for (var i = 0; i < 20; i++)
        {
            Assert.Equal(3, sampler.Sample(Logits));
        }
    }

    /// <summary>
    /// Tests that the repetition penalty steers selection away from already-seen tokens.
    /// </summary>
    [Fact]
    public void Sample_WithRepetitionPenalty_ShouldAvoidPenalizedArgMax()
    {
        // With temperature 0 the pipeline is argmax; penalizing the top token (index 3) shifts the
        // deterministic choice to the next-highest positive logit (index 1, 0.5).
        var sampler = new CombinedSampling(temperature: 0.0f, topK: null, topP: null, repetitionPenalty: 2.0f);

        var result = sampler.Sample(Logits, contextTokens: [3]);

        Assert.Equal(1, result);
    }

    /// <summary>
    /// Tests that the constructor rejects a negative temperature.
    /// </summary>
    [Fact]
    public void Constructor_WithNegativeTemperature_ShouldThrowArgumentOutOfRangeException()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => new CombinedSampling(temperature: -0.1f, topK: null, topP: null));
    }

    /// <summary>
    /// Tests that the constructor rejects a non-positive top-k.
    /// </summary>
    [Fact]
    public void Constructor_WithInvalidTopK_ShouldThrowArgumentOutOfRangeException()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => new CombinedSampling(temperature: 1.0f, topK: 0, topP: null));
    }

    /// <summary>
    /// Tests that the constructor rejects a top-p outside the (0, 1] range.
    /// </summary>
    [Fact]
    public void Constructor_WithInvalidTopP_ShouldThrowArgumentOutOfRangeException()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => new CombinedSampling(temperature: 1.0f, topK: null, topP: 0.0f));
        Assert.Throws<ArgumentOutOfRangeException>(() => new CombinedSampling(temperature: 1.0f, topK: null, topP: 1.1f));
    }

    /// <summary>
    /// Tests that the constructor rejects a non-positive repetition penalty.
    /// </summary>
    [Fact]
    public void Constructor_WithInvalidRepetitionPenalty_ShouldThrowArgumentOutOfRangeException()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => new CombinedSampling(temperature: 1.0f, topK: null, topP: null, repetitionPenalty: 0.0f));
    }

    /// <summary>
    /// Tests that sampling from an empty logit list throws.
    /// </summary>
    [Fact]
    public void Sample_WithEmptyLogits_ShouldThrowArgumentException()
    {
        var sampler = new CombinedSampling(temperature: 1.0f, topK: 3, topP: 0.9f, seed: 42);

        Assert.Throws<ArgumentException>(() => sampler.Sample([]));
    }
}
