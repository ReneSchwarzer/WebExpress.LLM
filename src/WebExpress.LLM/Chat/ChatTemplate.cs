using System;
using System.Collections;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
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
/// loads the raw template content and implements the corresponding formatting logic in C#,
/// covering the complete Gemma-4 chat protocol: system/developer instructions, thinking mode,
/// tool declarations, tool calls, tool responses, multi-modal placeholders, and multi-turn
/// continuation. See <c>docs/CHAT_TEMPLATE.md</c> for the protocol reference.
/// </remarks>
public sealed class ChatTemplate
{
    /// <summary>
    /// The default file name for the chat template within a model directory.
    /// </summary>
    public const string DefaultFileName = "chat_template.jinja";

    private const string QuoteDelimiter = "<|\"|>";

    private static readonly HashSet<string> SchemaKeywordKeys =
        new(StringComparer.Ordinal) { "description", "type", "properties", "required", "nullable" };

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
    /// Formats the given chat messages into a prompt string according to the Gemma-4 chat protocol.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This is a faithful C# port of the reference <c>chat_template.jinja</c>. The emitted structure is:
    /// </para>
    /// <list type="number">
    ///   <item>The <see cref="BosToken"/>.</item>
    ///   <item>
    ///     A leading <c>system</c> turn when <paramref name="enableThinking"/> is set, when
    ///     <paramref name="tools"/> are supplied, or when the first message is a
    ///     <c>system</c>/<c>developer</c> message. The turn may contain, in order, the
    ///     <c>&lt;|think|&gt;</c> marker, the system content, and one <c>&lt;|tool&gt;…&lt;tool|&gt;</c>
    ///     block per tool.
    ///   </item>
    ///   <item>
    ///     One turn per remaining message. The <c>assistant</c> role maps to <c>model</c>; consecutive
    ///     assistant messages continue the same model turn. Each turn may include a reasoning channel,
    ///     tool calls, tool responses, and text or multi-modal content.
    ///   </item>
    ///   <item>
    ///     A trailing <c>&lt;|turn&gt;model\n</c> generation prompt when
    ///     <paramref name="addGenerationPrompt"/> is <see langword="true"/> and the last turn does not
    ///     leave an open tool call or tool response.
    ///   </item>
    /// </list>
    /// </remarks>
    /// <param name="messages">
    /// The list of chat messages to format. When null or empty, only the BOS token and optional
    /// generation prompt are emitted.
    /// </param>
    /// <param name="addGenerationPrompt">
    /// Whether to append a generation prompt at the end. Defaults to <see langword="true"/>.
    /// </param>
    /// <param name="tools">
    /// Optional tool/function declarations to advertise to the model in the leading system turn.
    /// </param>
    /// <param name="enableThinking">
    /// Whether to activate thinking mode by injecting the <c>&lt;|think|&gt;</c> marker into the
    /// leading system turn.
    /// </param>
    /// <returns>
    /// A formatted prompt string ready for tokenization and inference.
    /// </returns>
    public string ApplyTemplate(
        IReadOnlyList<ChatMessage> messages,
        bool addGenerationPrompt = true,
        IReadOnlyList<ToolDefinition> tools = null,
        bool enableThinking = false)
    {
        var builder = new StringBuilder();
        builder.Append(BosToken);

        // The reference template indexes messages[0] unconditionally; empty/null input is handled
        // here as a convenience so callers get the BOS token and an optional generation prompt.
        if (messages == null || messages.Count == 0)
        {
            if (addGenerationPrompt)
            {
                builder.Append("<|turn>model\n");
            }

            return builder.ToString();
        }

        var hasTools = tools != null && tools.Count > 0;
        var firstIsSystem = messages[0].Role is "system" or "developer";

        // Tracks the trailing token kind so the generation prompt and turn terminators match the
        // reference template. Mirrors the Jinja `ns.prev_message_type` variable.
        string previousMessageType = null;

        // Leading system / tool-definition block.
        if (enableThinking || hasTools || firstIsSystem)
        {
            builder.Append("<|turn>system\n");

            if (enableThinking)
            {
                builder.Append("<|think|>\n");
                previousMessageType = "think";
            }

            if (firstIsSystem)
            {
                builder.Append((messages[0].Content ?? string.Empty).Trim());
            }

            if (hasTools)
            {
                foreach (var tool in tools)
                {
                    builder.Append("<|tool>");
                    AppendFunctionDeclaration(builder, tool);
                    builder.Append("<tool|>");
                }

                previousMessageType = "tool";
            }

            builder.Append("<turn|>\n");
        }

        // When the first message was consumed as the system turn, the message loop starts after it.
        var start = firstIsSystem ? 1 : 0;
        var count = messages.Count - start;

        // Pre-scan for the last user message, used to guard reasoning-channel emission.
        var lastUserIndex = -1;
        for (var i = 0; i < count; i++)
        {
            if (messages[start + i].Role == "user")
            {
                lastUserIndex = i;
            }
        }

        for (var index = 0; index < count; index++)
        {
            var message = messages[start + index];

            // Tool messages are consumed by the forward-scan of the preceding assistant turn.
            if (message.Role == "tool")
            {
                continue;
            }

            previousMessageType = null;
            var role = message.Role == "assistant" ? "model" : message.Role;

            // Resolve the previous non-tool message to detect a continued model turn.
            string previousNonToolRole = null;
            for (var j = index - 1; j >= 0; j--)
            {
                if (messages[start + j].Role != "tool")
                {
                    previousNonToolRole = messages[start + j].Role;
                    break;
                }
            }

            var continueSameModelTurn = role == "model" && previousNonToolRole == "assistant";
            if (!continueSameModelTurn)
            {
                builder.Append("<|turn>").Append(role).Append('\n');
            }

            // Reasoning is only rendered for the current model turn that also issues tool calls.
            var hasToolCalls = message.ToolCalls != null && message.ToolCalls.Count > 0;
            if (!string.IsNullOrEmpty(message.Reasoning) && index > lastUserIndex && hasToolCalls)
            {
                builder.Append("<|channel>thought\n").Append(message.Reasoning).Append("\n<channel|>");
            }

            if (hasToolCalls)
            {
                foreach (var toolCall in message.ToolCalls)
                {
                    builder.Append("<|tool_call>call:").Append(toolCall.Name).Append('{');

                    if (toolCall.RawArguments != null)
                    {
                        builder.Append(toolCall.RawArguments);
                    }
                    else if (toolCall.Arguments != null)
                    {
                        var first = true;
                        foreach (var pair in DictSort(toolCall.Arguments))
                        {
                            if (!first)
                            {
                                builder.Append(',');
                            }

                            first = false;
                            builder.Append(pair.Key).Append(':');
                            AppendArgument(builder, pair.Value, escapeKeys: false);
                        }
                    }

                    builder.Append("}<tool_call|>");
                }

                previousMessageType = "tool_call";
            }

            // Tool responses: embedded on the assistant message (Gemma-native) or supplied as
            // subsequent role:tool messages (OpenAI Chat Completions), resolved via forward-scan.
            var toolResponseEmitted = false;
            if (message.ToolResponses != null && message.ToolResponses.Count > 0)
            {
                foreach (var toolResponse in message.ToolResponses)
                {
                    AppendToolResponseBlock(builder, toolResponse.Name ?? "unknown", toolResponse.Response);
                    toolResponseEmitted = true;
                    previousMessageType = "tool_response";
                }
            }
            else if (hasToolCalls)
            {
                for (var k = index + 1; k < count; k++)
                {
                    var follow = messages[start + k];
                    if (follow.Role != "tool")
                    {
                        break;
                    }

                    var name = follow.Name ?? "unknown";
                    if (follow.ToolCallId != null)
                    {
                        foreach (var toolCall in message.ToolCalls)
                        {
                            if (toolCall.Id != null && toolCall.Id == follow.ToolCallId)
                            {
                                name = toolCall.Name;
                            }
                        }
                    }

                    AppendToolResponseBlock(builder, name, ResolveToolMessageBody(follow));
                    toolResponseEmitted = true;
                    previousMessageType = "tool_response";
                }
            }

            AppendContent(builder, message, role);

            var contentIsTruthy = message.Parts != null
                ? message.Parts.Count > 0
                : message.Content != null && message.Content.Length > 0;

            if (previousMessageType == "tool_call" && !toolResponseEmitted)
            {
                builder.Append("<|tool_response>");
            }
            else if (!(toolResponseEmitted && !contentIsTruthy))
            {
                builder.Append("<turn|>\n");
            }
        }

        if (addGenerationPrompt && previousMessageType != "tool_response" && previousMessageType != "tool_call")
        {
            builder.Append("<|turn>model\n");
        }

        return builder.ToString();
    }

    /// <summary>
    /// Appends a message's content — either a plain string or a sequence of multi-modal parts. Model
    /// (assistant) text is passed through <see cref="StripThinking"/>; user/system text is trimmed.
    /// </summary>
    private static void AppendContent(StringBuilder builder, ChatMessage message, string role)
    {
        if (message.Parts != null)
        {
            foreach (var part in message.Parts)
            {
                switch (part.Type)
                {
                    case ContentPartType.Text:
                        var text = part.Text ?? string.Empty;
                        builder.Append(role == "model" ? StripThinking(text) : text.Trim());
                        break;
                    case ContentPartType.Image:
                        builder.Append("<|image|>");
                        break;
                    case ContentPartType.Audio:
                        builder.Append("<|audio|>");
                        break;
                    case ContentPartType.Video:
                        builder.Append("<|video|>");
                        break;
                }
            }
        }
        else if (message.Content != null)
        {
            builder.Append(role == "model" ? StripThinking(message.Content) : message.Content.Trim());
        }
    }

    /// <summary>
    /// Resolves the textual body of a <c>role: tool</c> message: a plain string, or the concatenation
    /// of the text parts of a multi-modal content sequence.
    /// </summary>
    private static object ResolveToolMessageBody(ChatMessage message)
    {
        if (message.Parts == null)
        {
            return message.Content;
        }

        var builder = new StringBuilder();
        foreach (var part in message.Parts)
        {
            if (part.Type == ContentPartType.Text)
            {
                builder.Append(part.Text ?? string.Empty);
            }
        }

        return builder.ToString();
    }

    /// <summary>
    /// Appends a <c>&lt;|tool_response&gt;…&lt;tool_response|&gt;</c> block for the given tool name and
    /// response payload.
    /// </summary>
    private static void AppendToolResponseBlock(StringBuilder builder, string toolName, object response)
    {
        builder.Append("<|tool_response>");

        if (response is IReadOnlyDictionary<string, object> map)
        {
            builder.Append("response:").Append(toolName).Append('{');
            var first = true;
            foreach (var pair in DictSort(map))
            {
                if (!first)
                {
                    builder.Append(',');
                }

                first = false;
                builder.Append(pair.Key).Append(':');
                AppendArgument(builder, pair.Value, escapeKeys: false);
            }

            builder.Append('}');
        }
        else
        {
            builder.Append("response:").Append(toolName).Append("{value:");
            AppendArgument(builder, response, escapeKeys: false);
            builder.Append('}');
        }

        builder.Append("<tool_response|>");
    }

    /// <summary>
    /// Appends a single tool/function declaration block (without the surrounding
    /// <c>&lt;|tool&gt;</c>/<c>&lt;tool|&gt;</c> markers).
    /// </summary>
    private static void AppendFunctionDeclaration(StringBuilder builder, ToolDefinition tool)
    {
        builder.Append("declaration:").Append(tool.Name).Append("{description:")
            .Append(QuoteDelimiter).Append(tool.Description).Append(QuoteDelimiter);

        var parameters = tool.Parameters;
        if (parameters != null && parameters.Count > 0)
        {
            builder.Append(",parameters:{");

            if (GetValue(parameters, "properties") is IReadOnlyDictionary<string, object> properties && properties.Count > 0)
            {
                builder.Append("properties:{");
                AppendParameters(builder, properties);
                builder.Append("},");
            }

            if (GetValue(parameters, "required") is IEnumerable required && IsTruthy(required))
            {
                builder.Append("required:[");
                AppendQuotedList(builder, required);
                builder.Append("],");
            }

            var parametersType = GetString(parameters, "type");
            if (!string.IsNullOrEmpty(parametersType))
            {
                builder.Append("type:").Append(QuoteDelimiter).Append(parametersType.ToUpperInvariant()).Append(QuoteDelimiter).Append('}');
            }
        }

        if (tool.Response != null)
        {
            builder.Append(",response:{");

            var description = GetString(tool.Response, "description");
            if (!string.IsNullOrEmpty(description))
            {
                builder.Append("description:").Append(QuoteDelimiter).Append(description).Append(QuoteDelimiter).Append(',');
            }

            var responseType = GetString(tool.Response, "type");
            if (string.Equals(responseType?.ToUpperInvariant(), "OBJECT", StringComparison.Ordinal))
            {
                builder.Append("type:").Append(QuoteDelimiter).Append("OBJECT").Append(QuoteDelimiter).Append('}');
            }
        }

        builder.Append('}');
    }

    /// <summary>
    /// Appends the properties of a JSON-Schema-shaped parameter map, mirroring the reference
    /// <c>format_parameters</c> macro (nested objects, arrays, enums, required, and nullable).
    /// </summary>
    private static void AppendParameters(StringBuilder builder, IReadOnlyDictionary<string, object> properties)
    {
        var foundFirst = false;
        foreach (var pair in DictSort(properties))
        {
            if (SchemaKeywordKeys.Contains(pair.Key))
            {
                continue;
            }

            if (foundFirst)
            {
                builder.Append(',');
            }

            foundFirst = true;

            var schema = pair.Value as IReadOnlyDictionary<string, object>;
            builder.Append(pair.Key).Append(":{");
            var addComma = false;

            var description = GetString(schema, "description");
            if (!string.IsNullOrEmpty(description))
            {
                builder.Append("description:").Append(QuoteDelimiter).Append(description).Append(QuoteDelimiter);
                addComma = true;
            }

            var typeUpper = GetString(schema, "type")?.ToUpperInvariant();

            if (typeUpper == "STRING")
            {
                var enumValue = GetValue(schema, "enum");
                if (IsTruthy(enumValue))
                {
                    addComma = AppendCommaIfNeeded(builder, addComma);
                    builder.Append("enum:");
                    AppendArgument(builder, enumValue, escapeKeys: true);
                }
            }
            else if (typeUpper == "ARRAY")
            {
                if (GetValue(schema, "items") is IReadOnlyDictionary<string, object> items && items.Count > 0)
                {
                    addComma = AppendCommaIfNeeded(builder, addComma);
                    builder.Append("items:{");
                    AppendArrayItems(builder, items);
                    builder.Append('}');
                }
            }

            if (IsTruthy(GetValue(schema, "nullable")))
            {
                addComma = AppendCommaIfNeeded(builder, addComma);
                builder.Append("nullable:true");
            }

            if (typeUpper == "OBJECT")
            {
                addComma = AppendCommaIfNeeded(builder, addComma);
                builder.Append("properties:{");
                if (GetValue(schema, "properties") is IReadOnlyDictionary<string, object> nested)
                {
                    AppendParameters(builder, nested);
                }
                else if (schema != null)
                {
                    AppendParameters(builder, schema);
                }

                builder.Append('}');

                if (GetValue(schema, "required") is IEnumerable required && IsTruthy(required))
                {
                    addComma = AppendCommaIfNeeded(builder, addComma);
                    builder.Append("required:[");
                    AppendQuotedList(builder, required);
                    builder.Append(']');
                }
            }

            AppendCommaIfNeeded(builder, addComma);
            builder.Append("type:").Append(QuoteDelimiter).Append(typeUpper).Append(QuoteDelimiter).Append('}');
        }
    }

    /// <summary>
    /// Appends the body of an array schema's <c>items</c> map, mirroring the reference macro's
    /// per-key handling of <c>properties</c>, <c>required</c>, and <c>type</c>.
    /// </summary>
    private static void AppendArrayItems(StringBuilder builder, IReadOnlyDictionary<string, object> items)
    {
        var foundFirst = false;
        foreach (var pair in DictSort(items))
        {
            if (pair.Value == null)
            {
                continue;
            }

            if (foundFirst)
            {
                builder.Append(',');
            }

            foundFirst = true;

            switch (pair.Key)
            {
                case "properties":
                    builder.Append("properties:{");
                    if (pair.Value is IReadOnlyDictionary<string, object> nested)
                    {
                        AppendParameters(builder, nested);
                    }

                    builder.Append('}');
                    break;
                case "required":
                    builder.Append("required:[");
                    AppendQuotedList(builder, pair.Value as IEnumerable);
                    builder.Append(']');
                    break;
                case "type":
                    builder.Append("type:");
                    if (pair.Value is string typeString)
                    {
                        AppendArgument(builder, typeString.ToUpperInvariant(), escapeKeys: true);
                    }
                    else if (pair.Value is IEnumerable typeSequence)
                    {
                        var uppercased = typeSequence.Cast<object>()
                            .Select(item => (object)item?.ToString()?.ToUpperInvariant())
                            .ToList();
                        AppendArgument(builder, uppercased, escapeKeys: true);
                    }

                    break;
                default:
                    builder.Append(pair.Key).Append(':');
                    AppendArgument(builder, pair.Value, escapeKeys: true);
                    break;
            }
        }
    }

    /// <summary>
    /// Appends a value in the model's argument syntax, mirroring the reference <c>format_argument</c>
    /// macro: strings are wrapped in <c>&lt;|"|&gt;</c> delimiters, booleans become <c>true</c>/
    /// <c>false</c>, maps and sequences recurse, and other scalars are emitted via their invariant
    /// string form.
    /// </summary>
    private static void AppendArgument(StringBuilder builder, object argument, bool escapeKeys)
    {
        switch (argument)
        {
            case string text:
                builder.Append(QuoteDelimiter).Append(text).Append(QuoteDelimiter);
                break;
            case bool boolean:
                builder.Append(boolean ? "true" : "false");
                break;
            case IReadOnlyDictionary<string, object> map:
                builder.Append('{');
                var firstEntry = true;
                foreach (var pair in DictSort(map))
                {
                    if (!firstEntry)
                    {
                        builder.Append(',');
                    }

                    firstEntry = false;
                    if (escapeKeys)
                    {
                        builder.Append(QuoteDelimiter).Append(pair.Key).Append(QuoteDelimiter);
                    }
                    else
                    {
                        builder.Append(pair.Key);
                    }

                    builder.Append(':');
                    AppendArgument(builder, pair.Value, escapeKeys);
                }

                builder.Append('}');
                break;
            case IEnumerable sequence:
                builder.Append('[');
                var firstItem = true;
                foreach (var item in sequence)
                {
                    if (!firstItem)
                    {
                        builder.Append(',');
                    }

                    firstItem = false;
                    AppendArgument(builder, item, escapeKeys);
                }

                builder.Append(']');
                break;
            default:
                builder.Append(FormatScalar(argument));
                break;
        }
    }

    /// <summary>
    /// Appends a comma before the next entry when a previous sibling entry was written, and reports
    /// that a sibling now exists. Mirrors the reference macros' <c>add_comma</c> bookkeeping.
    /// </summary>
    private static bool AppendCommaIfNeeded(StringBuilder builder, bool addComma)
    {
        if (addComma)
        {
            builder.Append(',');
        }

        return true;
    }

    /// <summary>
    /// Appends a comma-separated list of quote-delimited items, e.g. a JSON-Schema <c>required</c> array.
    /// </summary>
    private static void AppendQuotedList(StringBuilder builder, IEnumerable items)
    {
        if (items == null)
        {
            return;
        }

        var first = true;
        foreach (var item in items)
        {
            if (!first)
            {
                builder.Append(',');
            }

            first = false;
            builder.Append(QuoteDelimiter).Append(item).Append(QuoteDelimiter);
        }
    }

    /// <summary>
    /// Removes the model's generated thinking channel(s) from text. Everything between a
    /// <c>&lt;|channel&gt;</c> marker and the following <c>&lt;channel|&gt;</c> is dropped, and the
    /// result is trimmed. Mirrors the reference <c>strip_thinking</c> macro.
    /// </summary>
    private static string StripThinking(string text)
    {
        if (text == null)
        {
            return string.Empty;
        }

        var builder = new StringBuilder();
        foreach (var part in text.Split("<channel|>"))
        {
            var channelStart = part.IndexOf("<|channel>", StringComparison.Ordinal);
            builder.Append(channelStart >= 0 ? part.Substring(0, channelStart) : part);
        }

        return builder.ToString().Trim();
    }

    /// <summary>
    /// Orders the entries of a map by key using a case-insensitive, invariant, ordinal comparison —
    /// matching Jinja's default <c>dictsort</c> behavior. LINQ's stable ordering preserves the
    /// original order for keys that compare equal.
    /// </summary>
    private static IEnumerable<KeyValuePair<string, object>> DictSort(IReadOnlyDictionary<string, object> map)
    {
        return map.OrderBy(static pair => pair.Key.ToLowerInvariant(), StringComparer.Ordinal);
    }

    private static object GetValue(IReadOnlyDictionary<string, object> map, string key)
    {
        return map != null && map.TryGetValue(key, out var value) ? value : null;
    }

    private static string GetString(IReadOnlyDictionary<string, object> map, string key)
    {
        return GetValue(map, key) as string;
    }

    /// <summary>
    /// Evaluates the truthiness of a value using Python/Jinja semantics: null, <c>false</c>, empty
    /// strings, empty collections, and zero are falsy; everything else is truthy.
    /// </summary>
    private static bool IsTruthy(object value)
    {
        switch (value)
        {
            case null:
                return false;
            case bool boolean:
                return boolean;
            case string text:
                return text.Length > 0;
            case IReadOnlyDictionary<string, object> map:
                return map.Count > 0;
            case IEnumerable sequence:
                foreach (var _ in sequence)
                {
                    return true;
                }

                return false;
            case int i:
                return i != 0;
            case long l:
                return l != 0;
            case double d:
                return d != 0;
            case float f:
                return f != 0;
            default:
                return true;
        }
    }

    /// <summary>
    /// Formats a scalar (non-string, non-boolean, non-collection) value using the invariant culture,
    /// so numeric output is locale-independent.
    /// </summary>
    private static string FormatScalar(object value)
    {
        return value switch
        {
            null => "None",
            IFormattable formattable => formattable.ToString(null, CultureInfo.InvariantCulture),
            _ => value.ToString()
        };
    }
}
