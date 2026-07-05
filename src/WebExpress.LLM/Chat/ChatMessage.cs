using System.Collections.Generic;

namespace WebExpress.LLM.Chat;

/// <summary>
/// Represents a single message in a chat conversation, including the sender's role and the message content.
/// </summary>
/// <param name="Role">
/// The role of the message sender, such as "user", "assistant", "system", "developer", or "tool".
/// This value determines how the message is interpreted in the conversation context. Cannot be null.
/// </param>
/// <param name="Content">
/// The textual content of the chat message. May be <see langword="null"/> for messages that carry
/// only tool calls, tool responses, or multi-modal <see cref="Parts"/>.
/// </param>
public sealed record ChatMessage(string Role, string Content)
{
    /// <summary>
    /// Gets the multi-modal content parts of the message. When set, this ordered sequence supersedes
    /// <see cref="Content"/> and may interleave text with image, audio, and video placeholders.
    /// </summary>
    public IReadOnlyList<ContentPart> Parts { get; init; }

    /// <summary>
    /// Gets the model's reasoning/thinking text for this turn. Emitted as a
    /// <c>&lt;|channel&gt;thought…&lt;channel|&gt;</c> block only for the current model turn that also
    /// issues tool calls; otherwise it is omitted from the formatted prompt.
    /// </summary>
    public string Reasoning { get; init; }

    /// <summary>
    /// Gets the tool/function calls issued by the model in this turn, or <see langword="null"/> when
    /// there are none.
    /// </summary>
    public IReadOnlyList<ToolCall> ToolCalls { get; init; }

    /// <summary>
    /// Gets the tool responses embedded directly on this message (Gemma-native tool calling), or
    /// <see langword="null"/> when responses are provided as separate <c>role: tool</c> messages.
    /// </summary>
    public IReadOnlyList<ToolResponse> ToolResponses { get; init; }

    /// <summary>
    /// Gets the tool name for a <c>role: tool</c> response message. Used to label the emitted tool
    /// response when it cannot be resolved from a matching <see cref="ToolCall.Id"/>.
    /// </summary>
    public string Name { get; init; }

    /// <summary>
    /// Gets the identifier that correlates a <c>role: tool</c> response message with the originating
    /// <see cref="ToolCall.Id"/> (OpenAI-style tool calling).
    /// </summary>
    public string ToolCallId { get; init; }
}
