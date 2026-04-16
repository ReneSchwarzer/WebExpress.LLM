using System;
using System.Collections.Generic;
using System.Linq;
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
    /// <exception cref="ArgumentNullException">
    /// Thrown when <paramref name="tokenizer"/> or <paramref name="inferenceEngine"/> is null.
    /// </exception>
    public ChatSession(ITokenizer tokenizer, IInferenceEngine inferenceEngine)
    {
        _tokenizer = tokenizer ?? throw new ArgumentNullException(nameof(tokenizer));
        _inferenceEngine = inferenceEngine ?? throw new ArgumentNullException(nameof(inferenceEngine));
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

        var prompt = string.Join('\n', _messages.Select(static message => $"{message.Role}: {message.Content}"));
        var promptTokens = _tokenizer.Encode(prompt);
        var responseTokens = _inferenceEngine.GenerateTokens(promptTokens, maxNewTokens);
        var responseText = _tokenizer.Decode(responseTokens);

        var assistant = new ChatMessage("assistant", responseText);
        _messages.Add(assistant);

        return assistant;
    }
}
