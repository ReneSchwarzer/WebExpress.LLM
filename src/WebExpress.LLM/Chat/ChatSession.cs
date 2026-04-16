using WebExpress.LLM.Inference;
using WebExpress.LLM.Tokenization;

namespace WebExpress.LLM.Chat;

public sealed class ChatSession
{
    private readonly ITokenizer _tokenizer;
    private readonly IInferenceEngine _inferenceEngine;
    private readonly List<ChatMessage> _messages = new();

    public ChatSession(ITokenizer tokenizer, IInferenceEngine inferenceEngine)
    {
        _tokenizer = tokenizer ?? throw new ArgumentNullException(nameof(tokenizer));
        _inferenceEngine = inferenceEngine ?? throw new ArgumentNullException(nameof(inferenceEngine));
    }

    public IReadOnlyList<ChatMessage> Messages => _messages;

    public ChatMessage Send(string userMessage, int maxNewTokens = 32)
    {
        if (string.IsNullOrWhiteSpace(userMessage))
        {
            throw new ArgumentException("Message must be provided.", nameof(userMessage));
        }

        var user = new ChatMessage("user", userMessage);
        _messages.Add(user);

        var prompt = string.Join('\n', _messages.Select(static message => $"{message.Role}: {message.Content}"));
        var promptTokens = _tokenizer.Encode(prompt);
        var responseTokens = _inferenceEngine.GenerateTokens(promptTokens, maxNewTokens);
        var responseText = _tokenizer.Decode(responseTokens);

        var assistant = new ChatMessage("assistant", responseText);
        _messages.Add(assistant);

        return assistant;
    }
}
