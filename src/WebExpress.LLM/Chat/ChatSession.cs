using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using WebExpress.LLM.Inference;
using WebExpress.LLM.Tokenization;

namespace WebExpress.LLM.Chat;

/// <summary>
/// Represents a chat session that manages the exchange of messages between a user and an assistant, maintaining
/// conversation history and generating assistant responses.
/// </summary>
public sealed class ChatSession
{
    private readonly ITokenizer _tokenizer;
    private readonly IInferenceEngine _inferenceEngine;
    private readonly ChatTemplate _chatTemplate;
    private readonly List<ChatMessage> _messages = [];

    /// <summary>
    /// Gets the collection of chat messages in the conversation.
    /// </summary>
    public IReadOnlyList<ChatMessage> Messages => _messages;

    /// <summary>
    /// Initializes a new instance of the ChatSession class with the specified tokenizer  
    /// and inference engine.
    /// </summary>
    /// <param name="tokenizer">
    /// The ITokenizer instance used to tokenize text input within the session.  
    /// Must not be null.
    /// </param>
    /// <param name="inferenceEngine">
    /// The IInferenceEngine instance used to process input and generate responses  
    /// within the session. Must not be null.
    /// </param>
    /// <param name="chatTemplate">
    /// An optional <see cref="ChatTemplate"/> used to format conversation messages into
    /// model-specific prompt strings. When <see langword="null"/>, a simple
    /// <c>role: content</c> format is used.
    /// </param>
    /// <exception cref="ArgumentNullException">
    /// Thrown when <paramref name="tokenizer"/> or <paramref name="inferenceEngine"/> is null.
    /// </exception>
    public ChatSession(ITokenizer tokenizer, IInferenceEngine inferenceEngine, ChatTemplate chatTemplate = null)
    {
        _tokenizer = tokenizer ?? throw new ArgumentNullException(nameof(tokenizer));
        _inferenceEngine = inferenceEngine ?? throw new ArgumentNullException(nameof(inferenceEngine));
        _chatTemplate = chatTemplate;
    }

    /// <summary>
    /// Sends a user message to the chat and generates a response from the assistant.
    /// </summary>
    /// <param name="userMessage">
    /// The message content to send as the user. Cannot be null, empty, or consist only of
    /// white-space characters.
    /// </param>
    /// <param name="maxNewTokens">
    /// The maximum number of tokens to generate for the assistant's response. Must be a positive
    /// integer. The default is 32.</param>
    /// <returns>
    /// A ChatMessage representing the assistant's response to the user message.
    /// </returns>
    /// <exception cref="ArgumentException">
    /// Thrown if userMessage is null, empty, or consists only of white-space characters.
    /// </exception>
    public ChatMessage Send(string userMessage, int maxNewTokens = 32)
    {
        if (string.IsNullOrWhiteSpace(userMessage))
        {
            throw new ArgumentException("Message must be provided.", nameof(userMessage));
        }

        var user = new ChatMessage("user", userMessage);
        _messages.Add(user);

        var prompt = FormatPrompt();
        var promptTokens = _tokenizer.Encode(prompt);
        var responseTokens = _inferenceEngine.GenerateTokens(promptTokens, maxNewTokens);
        var responseText = _tokenizer.Decode(responseTokens);

        var assistant = new ChatMessage("assistant", responseText);
        _messages.Add(assistant);

        return assistant;
    }

    /// <summary>
    /// Asynchronously sends a user message and streams the assistant's response token by token.
    /// </summary>
    /// <param name="userMessage">
    /// The message content to send as the user. Cannot be null, empty, or consist only of
    /// white-space characters.
    /// </param>
    /// <param name="maxNewTokens">
    /// The maximum number of tokens to generate for the assistant's response. Must be a positive
    /// integer. The default is 32.
    /// </param>
    /// <returns>
    /// An async enumerable that yields the assistant's response text incrementally as tokens are generated.
    /// </returns>
    /// <exception cref="ArgumentException">
    /// Thrown if userMessage is null, empty, or consists only of white-space characters.
    /// </exception>
    public async IAsyncEnumerable<string> SendAsync(string userMessage, int maxNewTokens = 32)
    {
        if (string.IsNullOrWhiteSpace(userMessage))
        {
            throw new ArgumentException("Message must be provided.", nameof(userMessage));
        }

        var user = new ChatMessage("user", userMessage);
        _messages.Add(user);

        var prompt = FormatPrompt();
        var promptTokens = _tokenizer.Encode(prompt);

        yield return "\n";
        yield return $"chat template: '{prompt}'\n";
        yield return $"prompt tokens: '[{string.Join(",", promptTokens)}]'\n";

        var responseBuilder = new StringBuilder();
        var responseTokens = new List<int>();

        await foreach (var token in _inferenceEngine.GenerateTokensAsync(promptTokens, maxNewTokens))
        {
            responseTokens.Add(token);
            var decodedText = _tokenizer.Decode([token]);
            responseBuilder.Append(decodedText);
            yield return decodedText.Trim() + " ";
        }

        var assistant = new ChatMessage("assistant", responseBuilder.ToString());
        _messages.Add(assistant);
    }

    /// <summary>
    /// Formats the current conversation history into a prompt string suitable for the inference engine.
    /// </summary>
    /// <remarks>
    /// When a <see cref="ChatTemplate"/> is available, the template's turn-based format is used.
    /// Otherwise, messages are concatenated using a simple <c>role: content</c> format separated
    /// by newlines.
    /// </remarks>
    /// <returns>
    /// A formatted prompt string ready for tokenization.
    /// </returns>
    private string FormatPrompt()
    {
        if (_chatTemplate != null)
        {
            return _chatTemplate.ApplyTemplate(_messages, addGenerationPrompt: true);
        }

        return string.Join('\n', _messages.Select(static message => $"{message.Role}: {message.Content}"));
    }
}
