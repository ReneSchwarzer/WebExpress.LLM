using WebExpress.LLM.Chat;
using WebExpress.LLM.Inference;
using WebExpress.LLM.Tokenization;

namespace WebExpress.LLM.Test.Chat;

/// <summary>
/// Provides unit tests for the ChatSession class, ensuring correct conversation tracking and message processing.
/// </summary>
public sealed class UnitTestChatSession
{
    /// <summary>
    /// Tests that sending a message tracks the conversation history and returns the assistant message.
    /// </summary>
    [Fact]
    public void Send_ShouldTrackConversationAndReturnAssistantMessage()
    {
        var tokenizer = new MockTokenizer();
        var inference = new MockInferenceEngine();
        var session = new ChatSession(tokenizer, inference);

        var response = session.Send("Hello", maxNewTokens: 3);

        Assert.Equal("assistant", response.Role);
        Assert.Equal("ABC", response.Content);
        Assert.Collection(
            session.Messages,
            message =>
            {
                Assert.Equal("user", message.Role);
                Assert.Equal("Hello", message.Content);
            },
            message =>
            {
                Assert.Equal("assistant", message.Role);
                Assert.Equal("ABC", message.Content);
            });
    }

    /// <summary>
    /// Tests that sending a message without a chat template uses the simple role:content format.
    /// </summary>
    [Fact]
    public void Send_WithoutTemplate_ShouldUseSimpleFormat()
    {
        var tokenizer = new CapturingTokenizer();
        var inference = new MockInferenceEngine();
        var session = new ChatSession(tokenizer, inference);

        session.Send("Hello", maxNewTokens: 3);

        Assert.Equal("user: Hello", tokenizer.LastEncodedText);
    }

    /// <summary>
    /// Tests that sending a message with a chat template uses the turn-based format.
    /// </summary>
    [Fact]
    public void Send_WithTemplate_ShouldUseTurnBasedFormat()
    {
        var tokenizer = new CapturingTokenizer();
        var inference = new MockInferenceEngine();
        var template = new ChatTemplate("{# template #}");
        var session = new ChatSession(tokenizer, inference, template);

        session.Send("Hello", maxNewTokens: 3);

        Assert.StartsWith("<bos>", tokenizer.LastEncodedText);
        Assert.Contains("<|turn>user\nHello<turn|>", tokenizer.LastEncodedText);
        Assert.EndsWith("<|turn>model\n", tokenizer.LastEncodedText);
    }

    /// <summary>
    /// Tests that the chat template properly maps assistant role to model in multi-turn conversations.
    /// </summary>
    [Fact]
    public void Send_WithTemplate_MultiTurn_ShouldMapAssistantToModel()
    {
        var tokenizer = new CapturingTokenizer();
        var inference = new MockInferenceEngine();
        var template = new ChatTemplate("{# template #}");
        var session = new ChatSession(tokenizer, inference, template);

        session.Send("First", maxNewTokens: 3);
        session.Send("Second", maxNewTokens: 3);

        // The prompt for the second message should contain both the first exchange and the second user message
        Assert.Contains("<|turn>model\nABC<turn|>", tokenizer.LastEncodedText);
        Assert.Contains("<|turn>user\nSecond<turn|>", tokenizer.LastEncodedText);
    }

    /// <summary>
    /// Tests that the ChatSession constructor accepts a null chat template without throwing.
    /// </summary>
    [Fact]
    public void Constructor_WithNullTemplate_ShouldNotThrow()
    {
        var tokenizer = new MockTokenizer();
        var inference = new MockInferenceEngine();

        var session = new ChatSession(tokenizer, inference, chatTemplate: null);

        Assert.NotNull(session);
    }

    private sealed class MockTokenizer : ITokenizer
    {
        public IReadOnlyList<int> Encode(string text) => text.Select(static character => (int)character).ToArray();

        public string Decode(IEnumerable<int> tokens) => new(tokens.Select(static token => (char)token).ToArray());
    }

    /// <summary>
    /// A tokenizer that captures the last encoded text for assertion purposes.
    /// </summary>
    private sealed class CapturingTokenizer : ITokenizer
    {
        public string? LastEncodedText { get; private set; }

        public IReadOnlyList<int> Encode(string text)
        {
            LastEncodedText = text;
            return text.Select(static character => (int)character).ToArray();
        }

        public string Decode(IEnumerable<int> tokens) => new(tokens.Select(static token => (char)token).ToArray());
    }

    private sealed class MockInferenceEngine : IInferenceEngine
    {
        public IReadOnlyList<int> GenerateTokens(IReadOnlyList<int> promptTokens, int maxNewTokens) => [65, 66, 67];

        public async IAsyncEnumerable<int> GenerateTokensAsync(IReadOnlyList<int> promptTokens, int maxNewTokens)
        {
            await Task.CompletedTask;
            yield return 65;
            yield return 66;
            yield return 67;
        }
    }
}
