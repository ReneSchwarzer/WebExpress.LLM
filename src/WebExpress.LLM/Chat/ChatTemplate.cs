using System;
using System.Collections.Generic;
using System.IO;
using System.Text;

namespace WebExpress.LLM.Chat;

/// <summary>
/// Represents a chat template loaded from a Jinja2 template file within a model directory.
/// Provides functionality to format conversation messages into model-specific prompt strings
/// using the turn-based structure defined by the template.
/// </summary>
/// <remarks>
/// The template file (<c>chat_template.jinja</c>) defines how messages are formatted into
/// prompts using special tokens such as <c>&lt;|turn&gt;</c> and <c>&lt;turn|&gt;</c> for
/// turn boundaries, and <c>&lt;bos&gt;</c> for the beginning-of-sequence marker. This class
/// loads the raw template content and implements the corresponding formatting logic in C#.
/// </remarks>
public sealed class ChatTemplate
{
    /// <summary>
    /// The default file name for the chat template within a model directory.
    /// </summary>
    public const string DefaultFileName = "chat_template.jinja";

    /// <summary>
    /// Gets the raw Jinja2 template content loaded from the file.
    /// </summary>
    public string TemplateContent { get; }

    /// <summary>
    /// Gets the beginning-of-sequence token used at the start of every formatted prompt.
    /// </summary>
    /// <remarks>
    /// This value defaults to <c>&lt;bos&gt;</c> and can be overridden when constructing the instance
    /// to match the tokenizer's expected special token.
    /// </remarks>
    public string BosToken { get; init; } = "<bos>";

    /// <summary>
    /// Initializes a new instance of the <see cref="ChatTemplate"/> class with the specified
    /// raw template content.
    /// </summary>
    /// <param name="templateContent">
    /// The raw Jinja2 template content. Must not be null, empty, or consist only of white-space characters.
    /// </param>
    /// <exception cref="ArgumentException">
    /// Thrown when <paramref name="templateContent"/> is null, empty, or consists only of white-space characters.
    /// </exception>
    public ChatTemplate(string templateContent)
    {
        if (string.IsNullOrWhiteSpace(templateContent))
        {
            throw new ArgumentException("Template content must be provided.", nameof(templateContent));
        }

        TemplateContent = templateContent;
    }

    /// <summary>
    /// Loads a <see cref="ChatTemplate"/> from the specified file path.
    /// </summary>
    /// <param name="filePath">
    /// The path to the Jinja2 template file. Must not be null, empty, or consist only of white-space characters.
    /// </param>
    /// <returns>
    /// A new <see cref="ChatTemplate"/> instance containing the template content from the file.
    /// </returns>
    /// <exception cref="ArgumentException">
    /// Thrown when <paramref name="filePath"/> is null, empty, or consists only of white-space characters.
    /// </exception>
    /// <exception cref="FileNotFoundException">
    /// Thrown when the specified file does not exist.
    /// </exception>
    public static ChatTemplate FromFile(string filePath)
    {
        if (string.IsNullOrWhiteSpace(filePath))
        {
            throw new ArgumentException("File path must be provided.", nameof(filePath));
        }

        if (!File.Exists(filePath))
        {
            throw new FileNotFoundException("Chat template file was not found.", filePath);
        }

        var content = File.ReadAllText(filePath);

        return new ChatTemplate(content);
    }

    /// <summary>
    /// Formats the given chat messages into a prompt string according to the template's
    /// turn-based format structure.
    /// </summary>
    /// <remarks>
    /// <para>
    /// The formatting follows the Gemma-4 chat template conventions:
    /// </para>
    /// <list type="bullet">
    ///   <item>
    ///     The prompt begins with the <see cref="BosToken"/>.
    ///   </item>
    ///   <item>
    ///     System or developer messages at position 0 are emitted as
    ///     <c>&lt;|turn&gt;system\n…content…&lt;turn|&gt;\n</c>.
    ///   </item>
    ///   <item>
    ///     User messages are emitted as <c>&lt;|turn&gt;user\n…content…&lt;turn|&gt;\n</c>.
    ///   </item>
    ///   <item>
    ///     Assistant messages use the <c>model</c> role identifier:
    ///     <c>&lt;|turn&gt;model\n…content…&lt;turn|&gt;\n</c>.
    ///   </item>
    ///   <item>
    ///     When <paramref name="addGenerationPrompt"/> is <see langword="true"/>, a final
    ///     <c>&lt;|turn&gt;model\n</c> marker is appended to prompt the model for a response.
    ///   </item>
    /// </list>
    /// </remarks>
    /// <param name="messages">
    /// The list of chat messages to format. If null or empty, only the BOS token and optional
    /// generation prompt are emitted.
    /// </param>
    /// <param name="addGenerationPrompt">
    /// Whether to append a generation prompt (<c>&lt;|turn&gt;model\n</c>) at the end.
    /// Defaults to <see langword="true"/>.
    /// </param>
    /// <returns>
    /// A formatted prompt string ready for tokenization and inference.
    /// </returns>
    public string ApplyTemplate(IReadOnlyList<ChatMessage> messages, bool addGenerationPrompt = true)
    {
        var builder = new StringBuilder();
        builder.Append(BosToken);

        if (messages == null || messages.Count == 0)
        {
            if (addGenerationPrompt)
            {
                builder.Append("<|turn>model\n");
            }

            return builder.ToString();
        }

        var startIndex = 0;

        // Handle system/developer message at position 0
        if (messages[0].Role is "system" or "developer")
        {
            builder.Append("<|turn>system\n");
            builder.Append(messages[0].Content.Trim());
            builder.Append("<turn|>\n");
            startIndex = 1;
        }

        // Process remaining messages
        for (var i = startIndex; i < messages.Count; i++)
        {
            var message = messages[i];
            var role = MapRole(message.Role);

            builder.Append("<|turn>");
            builder.Append(role);
            builder.Append('\n');

            if (message.Content != null)
            {
                // Model (assistant) messages are not trimmed to preserve the exact
                // output generated by the model, matching the Jinja2 template behavior
                // where model output passes through strip_thinking() without trim.
                // User and system messages are trimmed to remove incidental whitespace.
                builder.Append(role == "model" ? message.Content : message.Content.Trim());
            }

            builder.Append("<turn|>\n");
        }

        if (addGenerationPrompt)
        {
            builder.Append("<|turn>model\n");
        }

        return builder.ToString();
    }

    /// <summary>
    /// Maps an external role name to the model-internal role identifier.
    /// </summary>
    /// <param name="role">The external role name (e.g. "assistant", "user", "system").</param>
    /// <returns>
    /// The model-internal role identifier. "assistant" is mapped to "model"; all other
    /// values are returned unchanged.
    /// </returns>
    private static string MapRole(string role)
    {
        return role == "assistant" ? "model" : role;
    }
}
