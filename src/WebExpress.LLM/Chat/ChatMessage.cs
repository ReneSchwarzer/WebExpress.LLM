namespace WebExpress.LLM.Chat;

/// <summary>
/// Represents a single message in a chat conversation, including the sender's role and the message content.
/// </summary>
/// <param name="Role">
/// The role of the message sender, such as "user", "assistant", or "system". This value determines how the message is
/// interpreted in the conversation context. Cannot be null.
/// </param>
/// <param name="Content">
/// The textual content of the chat message. Cannot be null.
/// </param>
public sealed record ChatMessage(string Role, string Content);
