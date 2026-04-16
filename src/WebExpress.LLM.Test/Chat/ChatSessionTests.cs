using WebExpress.LLM.Chat;
using WebExpress.LLM.Inference;
using WebExpress.LLM.Tokenization;

namespace WebExpress.LLM.Test.Chat;

public sealed class ChatSessionTests
{
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

    private sealed class MockTokenizer : ITokenizer
    {
        public IReadOnlyList<int> Encode(string text) => text.Select(static character => (int)character).ToArray();

        public string Decode(IEnumerable<int> tokens) => new(tokens.Select(static token => (char)token).ToArray());
    }

    private sealed class MockInferenceEngine : IInferenceEngine
    {
        public IReadOnlyList<int> GenerateTokens(IReadOnlyList<int> promptTokens, int maxNewTokens) => [65, 66, 67];
    }
}
